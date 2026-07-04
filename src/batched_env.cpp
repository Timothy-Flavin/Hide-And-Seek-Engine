#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>
#include <chrono>
#include <fstream>
#ifndef _WIN32
#include <sched.h>
#endif
#include "env_state.h"

#include <omp.h>

#ifndef ssize_t
#ifdef _WIN32
typedef ptrdiff_t ssize_t;
#endif
#endif

namespace py = pybind11;

// uint8 spatial-obs encoding: binary/one-hot layers store OBS_ON, altitude is
// quantized into a byte. Consumers recover values with x.float()/255.
static constexpr uint8_t OBS_ON = 255;
static inline uint8_t alt_to_u8(float a)
{
    a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a); // altitudes are in [0,1]
    return static_cast<uint8_t>(a * 255.0f + 0.5f);
}

class BatchedEnvironment
{
private:
    std::unique_ptr<EnvironmentArena> arena;
    // Spatial OBS is stored as uint8 to cut memory 4x (PPO on-GPU rollout and
    // DQN/SAC pinned replay buffers). All obs spatial channels are binary except
    // altitude: binary layers store OBS_ON (255) and altitude is quantized to a
    // byte, so the consumer casts the whole tensor with a single x.float()/255.
    // The STATE spatial tensor stays float32 (opt-in, not used by the benchmark).
    uint8_t *torch_obs_spatial_base = nullptr;
    float *torch_obs_internal_base = nullptr;
    float *torch_state_spatial_base = nullptr;
    float *torch_state_internal_base = nullptr;

    // Ego-centric viewing. When enabled the spatial obs tensor shared with
    // pytorch is a fixed ego_size x ego_size window centered on each agent
    // rather than the full HxW map. The full-map fog-of-war accumulation is
    // kept in an internal, reused buffer (ego_full_obs) and a single crop copy
    // per agent is written into the shared tensor each step.
    bool ego_view = false;
    int ego_size = 0;
    std::vector<uint8_t> ego_full_obs;
    // Accumulation target for the incremental spatial-obs writes. In the
    // default (non-ego) path this simply aliases torch_obs_spatial_base so
    // behavior is unchanged and writes stay zero-copy. In ego mode it points
    // at ego_full_obs so the shared tensor can hold the smaller crop.
    uint8_t *obs_accum_base = nullptr;

    Mode mode;

    std::unique_ptr<DecentralizedPartialObsStrides> decentral_obs_strides;
    std::unique_ptr<CentralizedPartialObsStrides> central_obs_strides;
    std::unique_ptr<CentralizedStateStrides> central_state_strides;

    int width;
    int height;
    bool requires_state;
    float *individual_rewards = nullptr;
    uint8_t *env_terminated = nullptr;
    uint8_t *env_truncated = nullptr;

    std::vector<float> padded_rewards;
    std::vector<uint8_t> padded_terminated;
    std::vector<uint8_t> padded_truncated;
    int padded_reward_stride;
    int padded_byte_stride;

    std::vector<EnvStateView> env_views;
    std::vector<int> type_map;
    std::vector<float> altitude_map;
    std::vector<float> agent_speed_map;
    std::vector<float> agent_base_view_ranges;
    std::vector<uint8_t> supports_walking;
    std::vector<uint8_t> supports_aquatic;
    std::vector<uint8_t> supports_flying;
    std::vector<uint8_t> is_blocking;
    std::vector<uint8_t> saveable_rules;
    std::vector<uint8_t> agent_flags;
    std::vector<float> agent_max_altitudes;
    unsigned long long steps_taken = 0;

    void process_agent_movement(int e, const float *act_data)
    {
        EnvStateView view = env_views[e];

        for (int a = 0; a < n_agents; ++a)
        {
            AgentState &agent = view.agents[a];
            if (agent.deployment_remaining <= 0.0f)
                continue;
            float dy = act_data[e * n_agents * 2 + a * 2];
            float dx = act_data[e * n_agents * 2 + a * 2 + 1];

            float sqmag = dy * dy + dx * dx;
            if (sqmag > 1.0f)
            {
                float invmag = 1.0f / std::sqrt(sqmag);
                dy *= invmag;
                dx *= invmag;
            }

            int cy = static_cast<int>(agent.y);
            int cx = static_cast<int>(agent.x);

            uint32_t current_tile_type = view.grid[cy * width + cx].get_type();
            float speed = view.agent_speeds[a * n_tiles + current_tile_type];
            float ny = agent.y + dy * speed;
            float nx = agent.x + dx * speed;

            ny = std::max(0.0f, std::min(static_cast<float>(height) - 0.001f, ny));
            nx = std::max(0.0f, std::min(static_cast<float>(width) - 0.001f, nx));

            int target_y = static_cast<int>(ny);
            int target_x = static_cast<int>(nx);

            Tile &target_tile = view.grid[target_y * width + target_x];

            bool can_enter = false;
            float max_alt = fp16_to_float(agent.max_alt_fp16);

            // Dynamic Movement Restrictions & Blockers evaluated in Precedence: Flying > Walking > Swimming
            if (agent.can_fly() && target_tile.is_flyable() && target_tile.altitude <= max_alt)
            {
                can_enter = true;
            }
            else if (agent.can_walk() && target_tile.is_walkable())
            {
                can_enter = true;
            }
            else if (agent.can_swim() && target_tile.is_aquatic())
            {
                can_enter = true;
            }

            if (!can_enter || target_tile.is_blocking())
            {
                if (target_y != cy)
                {
                    ny = (dy > 0) ? target_y - 0.001f : target_y + 1.001f;
                }
                if (target_x != cx)
                {
                    nx = (dx > 0) ? target_x - 0.001f : target_x + 1.001f;
                }
            }

            agent.y = ny;
            agent.x = nx;

            int final_y = static_cast<int>(ny);
            int final_x = static_cast<int>(nx);
            uint16_t t_id = static_cast<uint16_t>(final_y * width + final_x);
            float final_altitude = view.grid[t_id].altitude;

            // Adjust View Range based on flight status
            if (agent.can_fly())
            {
                agent.view_range = std::max(agent_base_view_ranges[a] * max_alt, 1.0f);
            }
            else
            {
                agent.view_range = std::max(agent_base_view_ranges[a] * final_altitude, 1.0f);
            }
        }
    }

    template <Mode M, bool ReqState>
    void resolve_local_interactions_impl(int e)
    {
        EnvStateView view = env_views[e];
        const float RESCUE_DIST_SQ = 1.0f * 1.0f;

        for (int a = 0; a < n_agents; ++a)
        {
            AgentState &agent = view.agents[a];
            if (agent.deployment_remaining <= 0.0)
                continue;
            float vr = agent.view_range;
            float vr_sq = vr * vr;

            int center_x = static_cast<int>(agent.x);
            int center_y = static_cast<int>(agent.y);
            int min_y = std::max(0, static_cast<int>(agent.y - vr));
            int max_y = std::min(height - 1, static_cast<int>(agent.y + vr));
            int min_x = std::max(0, static_cast<int>(agent.x - vr));
            int max_x = std::min(width - 1, static_cast<int>(agent.x + vr));

            for (int y = min_y; y <= max_y; ++y)
            {
                float ysq = (agent.y - y) * (agent.y - y);
                for (int x = min_x; x <= max_x; ++x)
                {
                    float dist_sq = ysq + (agent.x - x) * (agent.x - x);
                    if (dist_sq <= vr_sq)
                    {
                        int t_idx = y * width + x;
                        Tile &t = view.grid[t_idx];
                        bool needs_global = !t.is_global_observed();
                        bool needs_local = false;
                        if constexpr (M == Mode::DECENTRALIZED)
                        {
                            needs_local = !t.has_agent_seen(a);
                        }
                        if (!needs_global && !needs_local)
                        {
                            continue;
                        }
                        bool is_occluded = false;
                        int x0 = center_x;
                        int y0 = center_y;

                        int dx = std::abs(x - x0), sx = x0 < x ? 1 : -1;
                        int dy = std::abs(y - y0), sy = y0 < y ? 1 : -1;
                        int err = dx - dy, e2;

                        int trace_idx = y0 * width + x0;
                        int step_x = sx;
                        int step_y = sy * width;

                        while (true)
                        {
                            if (trace_idx == t_idx)
                                break;

                            if (view.grid[trace_idx].is_blocking())
                            {
                                is_occluded = true;
                                break;
                            }

                            e2 = 2 * err;
                            if (e2 > -dy)
                            {
                                err -= dy;
                                trace_idx += step_x;
                            }
                            if (e2 < dx)
                            {
                                err += dx;
                                trace_idx += step_y;
                            }
                        }
                        if (is_occluded)
                            continue;

                        bool newly_global = false;
                        if (!t.is_global_observed())
                        {
                            t.set_global_observed();
                            (*view.undiscovered_remaining)--;
                            padded_rewards[e * padded_reward_stride + a] += reward_new_tile;
                            newly_global = true;
                        }

                        if constexpr (M == Mode::DECENTRALIZED)
                        {
                            if (!t.has_agent_seen(a))
                            {
                                t.set_agent_seen(a);
                                if (obs_accum_base != nullptr)
                                {
                                    ssize_t map_area = width * height;
                                    ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
                                    uint8_t *spat_base = obs_accum_base + e * (n_agents * spatial_stride) + a * spatial_stride;
                                    spat_base[decentral_obs_strides->TYLE_TYPE_START + t.get_type() * map_area + t_idx] = OBS_ON;
                                    spat_base[decentral_obs_strides->ALTITUDE_TYPE_START + t_idx] = alt_to_u8(t.altitude);
                                    spat_base[decentral_obs_strides->OBSERVED_START + t_idx] = OBS_ON;
                                }
                            }
                        }
                        else if constexpr (M == Mode::CENTRALIZED)
                        {
                            if (newly_global)
                            {
                                if (obs_accum_base != nullptr)
                                {
                                    ssize_t map_area = width * height;
                                    ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
                                    uint8_t *spat_base = obs_accum_base + e * spatial_stride;
                                    spat_base[central_obs_strides->TYLE_TYPE_START + t.get_type() * map_area + t_idx] = OBS_ON;
                                    spat_base[central_obs_strides->ALTITUDE_TYPE_START + t_idx] = alt_to_u8(t.altitude);
                                    spat_base[central_obs_strides->OBSERVED_START + t_idx] = OBS_ON;
                                }
                                if constexpr (ReqState)
                                {
                                    if (torch_state_spatial_base != nullptr)
                                    {
                                        ssize_t map_area = width * height;
                                        ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
                                        float *spat_base = torch_state_spatial_base + e * spatial_stride;
                                        spat_base[central_state_strides->TYLE_TYPE_START + t.get_type() * map_area + t_idx] = 1.0f;
                                        spat_base[central_state_strides->ALTITUDE_TYPE_START + t_idx] = t.altitude;
                                        spat_base[central_state_strides->OBSERVED_START + t_idx] = 1.0f;
                                    }
                                }
                            }
                        }
                        else if constexpr (M == Mode::NO_OBS)
                        {
                            if constexpr (ReqState)
                            {
                                if (newly_global && torch_state_spatial_base != nullptr)
                                {
                                    ssize_t map_area = width * height;
                                    ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
                                    float *spat_base = torch_state_spatial_base + e * spatial_stride;
                                    spat_base[central_state_strides->TYLE_TYPE_START + t.get_type() * map_area + t_idx] = 1.0f;
                                    spat_base[central_state_strides->ALTITUDE_TYPE_START + t_idx] = t.altitude;
                                    spat_base[central_state_strides->OBSERVED_START + t_idx] = 1.0f;
                                }
                            }
                        }
                    }
                }
            }

            for (int p = 0; p < n_pois; ++p)
            {
                POIState &poi = view.pois[p];
                if (poi.is_saved())
                    continue;

                float dist_sq = (agent.y - poi.y) * (agent.y - poi.y) + (agent.x - poi.x) * (agent.x - poi.x);

                if (dist_sq <= vr_sq)
                {
                    if (!poi.is_found())
                    {
                        poi.set_found(true);
                        padded_rewards[e * padded_reward_stride + a] += reward_found;
                    }

                    if (dist_sq <= RESCUE_DIST_SQ && poi.is_savable_by(a))
                    {
                        poi.set_saved(true);
                        padded_rewards[e * padded_reward_stride + a] += reward_saved;
                    }

                    if constexpr (M == Mode::DECENTRALIZED)
                    {
                        POIKnowledge &pk = view.poi_knowledge[a * n_pois + p];
                        pk.x = poi.x;
                        pk.y = poi.y;
                        pk.knows_found = poi.is_found();
                        pk.knows_saved = poi.is_saved();
                    }
                }
            }

            if constexpr (M == Mode::DECENTRALIZED)
            {
                AgentKnowledge &ak = view.agent_knowledge[a * n_agents + a];
                ak.x = agent.x;
                ak.y = agent.y;
                ak.has_contact = 1;
                ak.has_encountered = 1;
            }
        }
    }

    void resolve_local_interactions(int e)
    {
        if (mode == Mode::DECENTRALIZED)
        {
            if (requires_state)
                resolve_local_interactions_impl<Mode::DECENTRALIZED, true>(e);
            else
                resolve_local_interactions_impl<Mode::DECENTRALIZED, false>(e);
        }
        else if (mode == Mode::CENTRALIZED)
        {
            if (requires_state)
                resolve_local_interactions_impl<Mode::CENTRALIZED, true>(e);
            else
                resolve_local_interactions_impl<Mode::CENTRALIZED, false>(e);
        }
        else
        {
            if (requires_state)
                resolve_local_interactions_impl<Mode::NO_OBS, true>(e);
            else
                resolve_local_interactions_impl<Mode::NO_OBS, false>(e);
        }
    }

    void execute_radio(int e, const int *radio_act_data)
    {
        if (mode != Mode::DECENTRALIZED)
            return;

        EnvStateView view = env_views[e];

        for (int a = 0; a < n_agents; ++a)
        {
            for (int ai = 0; ai < n_agents; ++ai)
                view.agent_knowledge[ai * n_agents + a].has_contact = 0;

            int target_agent = radio_act_data[e * n_agents + a];

            if (target_agent == 0)
                continue;
            if (target_agent <= a)
                --target_agent;

            AgentKnowledge &target_ak = view.agent_knowledge[target_agent * n_agents + a];
            target_ak.x = view.agents[a].x;
            target_ak.y = view.agents[a].y;
            target_ak.has_contact = 1;
            target_ak.has_encountered = 1;

            for (int p = 0; p < n_pois; ++p)
            {
                POIKnowledge &sender_pk = view.poi_knowledge[a * n_pois + p];
                POIKnowledge &receiver_pk = view.poi_knowledge[target_agent * n_pois + p];

                if (sender_pk.knows_found && !receiver_pk.knows_found)
                {
                    receiver_pk.x = sender_pk.x;
                    receiver_pk.y = sender_pk.y;
                    receiver_pk.knows_found = 1;
                }
                if (sender_pk.knows_saved)
                {
                    receiver_pk.knows_saved = 1;
                }
            }

            float vr = view.agents[a].view_range;
            float vr_sq = vr * vr;
            int min_y = std::max(0, static_cast<int>(view.agents[a].y - vr));
            int max_y = std::min(height - 1, static_cast<int>(view.agents[a].y + vr));
            int min_x = std::max(0, static_cast<int>(view.agents[a].x - vr));
            int max_x = std::min(width - 1, static_cast<int>(view.agents[a].x + vr));

            for (int y = min_y; y <= max_y; ++y)
            {
                float ysq = (view.agents[a].y - y) * (view.agents[a].y - y);
                for (int x = min_x; x <= max_x; ++x)
                {
                    float dist_sq = ysq + (view.agents[a].x - x) * (view.agents[a].x - x);
                    if (dist_sq <= vr_sq)
                    {
                        int t_idx = y * width + x;
                        Tile &t = view.grid[t_idx];
                        if (t.has_agent_seen(a) && !t.has_agent_seen(target_agent))
                        {
                            t.set_agent_seen(target_agent);
                            if (obs_accum_base != nullptr)
                            {
                                ssize_t map_area = width * height;
                                ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
                                uint8_t *spat_base = obs_accum_base + e * (n_agents * spatial_stride) + target_agent * spatial_stride;
                                spat_base[decentral_obs_strides->TYLE_TYPE_START + t.get_type() * map_area + t_idx] = OBS_ON;
                                spat_base[decentral_obs_strides->ALTITUDE_TYPE_START + t_idx] = alt_to_u8(t.altitude);
                                spat_base[decentral_obs_strides->OBSERVED_START + t_idx] = OBS_ON;
                            }
                        }
                    }
                }
            }
        }
    }

    void update_battery_and_counters(int e)
    {
        EnvStateView view = env_views[e];
        for (int a = 0; a < n_agents; ++a)
        {
            if (view.agents[a].deployment_remaining > 0.0f)
            {
                view.agents[a].deployment_remaining -= delta_time;
                if (view.agents[a].deployment_remaining <= 0.0f)
                {
                    --(*view.agents_left);
                }
            }
        }
        int poi_left = 0;
        for (int p = 0; p < n_pois; ++p)
        {
            if (!view.pois[p].is_saved())
                poi_left++;
        }
        *view.poi_left = poi_left;
    }

    void fill_torch_obs(int e)
    {
        if (!obs_accum_base || !torch_obs_internal_base)
            return;

        EnvStateView view = env_views[e];
        ssize_t map_area = width * height;

        if (mode == Mode::DECENTRALIZED)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
            ssize_t internal_stride = 6;

            for (int a = 0; a < n_agents; ++a)
            {
                uint8_t *spat_base = obs_accum_base + e * (n_agents * spatial_stride) + a * spatial_stride;
                float *int_base = torch_obs_internal_base + e * (n_agents * internal_stride) + a * internal_stride;

                for (int p = 0; p < n_pois; ++p)
                {
                    POIKnowledge &pk = view.poi_knowledge[a * n_pois + p];
                    spat_base[decentral_obs_strides->PIO_START + pk.last_y * width + pk.last_x] = 0;
                    if (pk.knows_found && !pk.knows_saved)
                    {
                        int py = static_cast<int>(pk.y);
                        int px = static_cast<int>(pk.x);
                        spat_base[decentral_obs_strides->PIO_START + py * width + px] = OBS_ON;
                        pk.last_y = py;
                        pk.last_x = px;
                    }
                }

                spat_base[decentral_obs_strides->MY_LOCATION_START + view.agents[a].last_y * width + view.agents[a].last_x] = 0;
                int my_y = static_cast<int>(view.agents[a].y);
                int my_x = static_cast<int>(view.agents[a].x);
                spat_base[decentral_obs_strides->MY_LOCATION_START + my_y * width + my_x] = OBS_ON;
                view.agents[a].last_y = my_y;
                view.agents[a].last_x = my_x;

                int other_idx = 0;
                for (int a2 = 0; a2 < n_agents; ++a2)
                {
                    if (a == a2)
                        continue;
                    AgentKnowledge &ak = view.agent_knowledge[a * n_agents + a2];
                    if (ak.has_contact)
                    {
                        int oy = static_cast<int>(ak.y);
                        int ox = static_cast<int>(ak.x);
                        spat_base[decentral_obs_strides->OTHER_LOCATIONS_START + other_idx * map_area + ak.last_y * width + ak.last_x] = 0;
                        spat_base[decentral_obs_strides->OTHER_LOCATIONS_START + other_idx * map_area + oy * width + ox] = OBS_ON;
                        ak.last_y = oy;
                        ak.last_x = ox;
                        ak.has_encountered = 1;
                    }
                    other_idx++;
                }

                int_base[0] = view.agents[a].y;
                int_base[1] = view.agents[a].x;
                int_base[2] = agent_max_batteries[a];
                int_base[3] = view.agents[a].view_range;
                int_base[4] = view.agents[a].deployment_remaining;
                int_base[5] = static_cast<float>(view.agents[a].flags & 1U); // expose 'stuck' bit to tensor layer
            }
        }
        else if (mode == Mode::CENTRALIZED)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
            uint8_t *spat_base = obs_accum_base + e * spatial_stride;
            float *int_base = torch_obs_internal_base + e * (n_agents * 6);

            for (int p = 0; p < n_pois; ++p)
            {
                spat_base[central_obs_strides->PIO_START + view.pois[p].last_y * width + view.pois[p].last_x] = 0;
                if (view.pois[p].is_found() && !view.pois[p].is_saved())
                {
                    int py = static_cast<int>(view.pois[p].y);
                    int px = static_cast<int>(view.pois[p].x);
                    spat_base[central_obs_strides->PIO_START + view.pois[p].last_y * width + view.pois[p].last_x] = 0;
                    spat_base[central_obs_strides->PIO_START + py * width + px] = OBS_ON;
                    view.pois[p].last_y = py;
                    view.pois[p].last_x = px;
                }
            }

            for (int a = 0; a < n_agents; ++a)
            {
                spat_base[central_obs_strides->AGENT_LOCATIONS_START + a * map_area + view.agents[a].last_y * width + view.agents[a].last_x] = 0;

                int ay = static_cast<int>(view.agents[a].y);
                int ax = static_cast<int>(view.agents[a].x);
                spat_base[central_obs_strides->AGENT_LOCATIONS_START + a * map_area + ay * width + ax] = OBS_ON;
                view.agents[a].last_y = ay;
                view.agents[a].last_x = ax;

                int_base[a * 6 + 0] = view.agents[a].y;
                int_base[a * 6 + 1] = view.agents[a].x;
                int_base[a * 6 + 2] = agent_max_batteries[a];
                int_base[a * 6 + 3] = view.agents[a].view_range;
                int_base[a * 6 + 4] = view.agents[a].deployment_remaining;
                int_base[a * 6 + 5] = static_cast<float>(view.agents[a].flags & 1U);
            }
        }
    }

    // Copy an ego_size x ego_size crop of the full-map accumulation buffer into
    // the shared pytorch tensor, centered on each agent. Out-of-bounds cells are
    // zero-padded. This is the single unavoidable copy from the internal global
    // state to the shared (smaller) ego tensor; no allocation happens here, all
    // buffers are pre-sized and reused. Called per-env from the omp step loop.
    void fill_ego_obs(int e)
    {
        if (!ego_view || !torch_obs_spatial_base || !obs_accum_base)
            return;

        EnvStateView view = env_views[e];
        const int S = ego_size;
        const int half = S / 2;
        const ssize_t map_area = static_cast<ssize_t>(width) * height;
        const ssize_t channels = n_tiles + 3 + n_agents;
        const ssize_t full_agent_stride = channels * map_area; // per-map slab in accum buffer
        const ssize_t ego_plane = static_cast<ssize_t>(S) * S;
        const ssize_t ego_agent_stride = channels * ego_plane; // per-agent slab in shared tensor

        for (int a = 0; a < n_agents; ++a)
        {
            // Decentralized: each agent has its own accumulated map.
            // Centralized: all agents crop from the single shared map.
            const uint8_t *src_map =
                (mode == Mode::DECENTRALIZED)
                    ? obs_accum_base + e * (n_agents * full_agent_stride) + a * full_agent_stride
                    : obs_accum_base + e * full_agent_stride;

            uint8_t *dst = torch_obs_spatial_base + e * (n_agents * ego_agent_stride) + a * ego_agent_stride;

            // Fully overwrite the agent's crop each step; zero handles OOB padding.
            std::memset(dst, 0, ego_agent_stride * sizeof(uint8_t));

            const int cy = static_cast<int>(view.agents[a].y);
            const int cx = static_cast<int>(view.agents[a].x);
            const int y0 = cy - half; // source row for destination row 0
            const int x0 = cx - half; // source col for destination col 0

            const int r0 = std::max(0, -y0);
            const int r1 = std::min(S, height - y0);
            const int c0 = std::max(0, -x0);
            const int c1 = std::min(S, width - x0);
            if (r1 <= r0 || c1 <= c0)
                continue; // agent fully outside map (should not happen) -> all zeros

            const int span = c1 - c0;
            for (ssize_t ch = 0; ch < channels; ++ch)
            {
                const uint8_t *src_ch = src_map + ch * map_area;
                uint8_t *dst_ch = dst + ch * ego_plane;
                for (int r = r0; r < r1; ++r)
                {
                    const uint8_t *src_row = src_ch + static_cast<ssize_t>(y0 + r) * width + (x0 + c0);
                    uint8_t *dst_row = dst_ch + static_cast<ssize_t>(r) * S + c0;
                    std::memcpy(dst_row, src_row, span * sizeof(uint8_t));
                }
            }
        }
    }

    void fill_torch_state(int e)
    {
        if (!torch_state_spatial_base || !torch_state_internal_base)
            return;

        EnvStateView view = env_views[e];
        ssize_t map_area = width * height;
        ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;

        float *spat_base = torch_state_spatial_base + e * spatial_stride;
        float *int_base = torch_state_internal_base + e * (n_agents * 6 + n_pois * 4);

        for (int p = 0; p < n_pois; ++p)
        {
            spat_base[central_state_strides->PIO_START + view.pois[p].last_y * width + view.pois[p].last_x] = 0.0f;
            if (!view.pois[p].is_saved())
            {
                int py = static_cast<int>(view.pois[p].y);
                int px = static_cast<int>(view.pois[p].x);
                if (py >= 0 && py < height && px >= 0 && px < width)
                {
                    spat_base[central_state_strides->PIO_START + py * width + px] = 1.0f;
                }
            }
        }

        int int_off = 0;
        for (int a = 0; a < n_agents; ++a)
        {
            spat_base[central_state_strides->AGENT_LOCATIONS_START + a * map_area + view.agents[a].last_y * width + view.agents[a].last_x] = 0.0f;
            int ay = static_cast<int>(view.agents[a].y);
            int ax = static_cast<int>(view.agents[a].x);
            if (ay >= 0 && ay < height && ax >= 0 && ax < width)
            {
                spat_base[central_state_strides->AGENT_LOCATIONS_START + a * map_area + ay * width + ax] = 1.0f;
            }
            int_base[int_off++] = view.agents[a].y;
            int_base[int_off++] = view.agents[a].x;
            int_base[int_off++] = agent_max_batteries[a];
            int_base[int_off++] = view.agents[a].view_range;
            int_base[int_off++] = view.agents[a].deployment_remaining;
            int_base[int_off++] = static_cast<float>(view.agents[a].flags & 1U);
        }

        for (int p = 0; p < n_pois; ++p)
        {
            int_base[int_off++] = view.pois[p].y;
            int_base[int_off++] = view.pois[p].x;
            int_base[int_off++] = static_cast<float>(view.pois[p].is_found());
            int_base[int_off++] = static_cast<float>(view.pois[p].is_saved());
        }
    }

public:
    int seed;
    int n_tiles;
    int n_pois;
    int n_agents;
    bool cooperative_rewards;
    float reward_new_tile;
    float reward_found;
    float reward_saved;
    float delta_time = 1.0f;
    int max_frames;
    int num_envs;
    int map_size;

    std::vector<std::mt19937> rngs;
    std::vector<int> current_frames;
    std::vector<float> init_agent_positions;
    std::vector<float> agent_max_batteries;
    std::vector<float> init_poi_positions;
    std::vector<std::string> radio_logs;
    std::vector<Tile> pristine_grid;

    BatchedEnvironment(
        int n_envs,
        int sim_seed,
        int w,
        int h,
        const std::vector<uint8_t> &supports_walk,
        const std::vector<uint8_t> &supports_aqua,
        const std::vector<uint8_t> &supports_fly,
        const std::vector<uint8_t> &is_block,
        const std::vector<int> &t_map,
        const std::vector<float> &alt_map,
        const std::vector<float> &speed_map,
        const std::vector<float> &agent_view_ranges,
        const std::vector<uint8_t> &saveable_rules,
        const std::vector<float> &initial_agent_pos,
        const std::vector<float> &initial_poi_pos,
        const std::vector<float> &agent_max_batteries_val,
        const std::vector<uint8_t> &agent_flags_val,
        const std::vector<float> &agent_max_altitudes_val,
        uintptr_t obs_spatial_ptr,
        uintptr_t obs_internal_ptr,
        uintptr_t state_spatial_ptr,
        uintptr_t state_internal_ptr,
        uintptr_t rewards_ptr,
        uintptr_t terminated_ptr,
        uintptr_t truncated_ptr,
        bool requires_state = false,
        bool coop_rewards = true,
        float reward_new_tile_val = 0.05f,
        float reward_found_val = 2.0f,
        float reward_saved_val = 20.0f,
        int max_frames_val = 250,
        int mode_value = 0,
        int init_mode = 0,
        bool ego_view_val = false,
        int ego_size_val = 0)
        : num_envs(n_envs), seed(sim_seed), width(w), height(h),
          supports_walking(std::move(supports_walk)), supports_aquatic(std::move(supports_aqua)),
          supports_flying(std::move(supports_fly)), is_blocking(std::move(is_block)),
          type_map(std::move(t_map)), altitude_map(std::move(alt_map)), agent_speed_map(std::move(speed_map)),
          agent_base_view_ranges(std::move(agent_view_ranges)),
          saveable_rules(std::move(saveable_rules)), init_agent_positions(std::move(initial_agent_pos)),
          init_poi_positions(std::move(initial_poi_pos)),
          agent_max_batteries(std::move(agent_max_batteries_val)),
          agent_flags(std::move(agent_flags_val)), agent_max_altitudes(std::move(agent_max_altitudes_val)),
          requires_state(requires_state), cooperative_rewards(coop_rewards),
          reward_new_tile(reward_new_tile_val), reward_found(reward_found_val), reward_saved(reward_saved_val),
          max_frames(max_frames_val), map_size(w * h), mode(static_cast<Mode>(mode_value))
    {
        if (mode_value < 0 || mode_value > 2)
            throw(std::invalid_argument("mode value < 0 or > 2 cannot be bound into Mode enum"));

        n_tiles = int(supports_walking.size());
        n_agents = int(init_agent_positions.size() / 2);
        n_pois = int(init_poi_positions.size() / 2);

        current_frames.resize(num_envs, 0);
        radio_logs.resize(num_envs, "");

        torch_obs_spatial_base = reinterpret_cast<uint8_t *>(obs_spatial_ptr);
        torch_obs_internal_base = reinterpret_cast<float *>(obs_internal_ptr);
        torch_state_spatial_base = reinterpret_cast<float *>(state_spatial_ptr);
        torch_state_internal_base = reinterpret_cast<float *>(state_internal_ptr);
        individual_rewards = reinterpret_cast<float *>(rewards_ptr);
        env_terminated = reinterpret_cast<uint8_t *>(terminated_ptr);
        env_truncated = reinterpret_cast<uint8_t *>(truncated_ptr);

        padded_reward_stride = std::max(64, static_cast<int>((n_agents * sizeof(float) + 255) / sizeof(float)));
        padded_byte_stride = 256;
        padded_rewards.resize(num_envs * padded_reward_stride, 0.0f);
        padded_terminated.resize(num_envs * padded_byte_stride, 0);
        padded_truncated.resize(num_envs * padded_byte_stride, 0);

        ssize_t spatial_channels = n_tiles + 3 + n_agents;
        ssize_t obs_spatial_stride = (mode == Mode::DECENTRALIZED) ? (n_agents * spatial_channels * map_size) : (spatial_channels * map_size);
        ssize_t obs_internal_stride = n_agents * 6;
        ssize_t state_spatial_stride = spatial_channels * map_size;
        ssize_t state_internal_stride = n_agents * 6 + n_pois * 4;

        // Resolve the accumulation target for incremental spatial-obs writes.
        // Default (non-ego): alias the shared pytorch tensor -> zero-copy, unchanged.
        // Ego: allocate a reused internal full-map buffer; the shared tensor holds
        // the per-agent ego crop instead of the whole map.
        ego_view = ego_view_val;
        ego_size = ego_size_val;
        if (ego_view)
        {
            if (ego_size <= 0)
                throw std::invalid_argument("ego_size must be > 0 when ego_view is enabled");
            if (mode != Mode::NO_OBS && torch_obs_spatial_base != nullptr)
            {
                // obs_spatial_stride is the per-env full-map size (matches the
                // tensor shape used in the non-ego path).
                ego_full_obs.assign(static_cast<size_t>(num_envs) * obs_spatial_stride, static_cast<uint8_t>(0));
                obs_accum_base = ego_full_obs.data();
            }
        }
        else
        {
            obs_accum_base = torch_obs_spatial_base;
        }

        if (init_mode == 0)
        {
#pragma omp parallel for schedule(static)
            for (int e = 0; e < num_envs; ++e)
            {
                if (obs_accum_base)
                    std::memset(obs_accum_base + e * obs_spatial_stride, 0, obs_spatial_stride * sizeof(uint8_t));
                if (torch_obs_internal_base)
                    std::memset(torch_obs_internal_base + e * obs_internal_stride, 0, obs_internal_stride * sizeof(float));
                if (torch_state_spatial_base && requires_state)
                    std::memset(torch_state_spatial_base + e * state_spatial_stride, 0, state_spatial_stride * sizeof(float));
                if (torch_state_internal_base && requires_state)
                    std::memset(torch_state_internal_base + e * state_internal_stride, 0, state_internal_stride * sizeof(float));
                if (individual_rewards)
                    std::memset(individual_rewards + e * n_agents, 0, n_agents * sizeof(float));
                if (env_terminated)
                    env_terminated[e] = 0;
                if (env_truncated)
                    env_truncated[e] = 0;
                padded_terminated[e * padded_byte_stride] = 0;
                padded_truncated[e * padded_byte_stride] = 0;
            }
        }
        else
        {
            for (int e = 0; e < num_envs; ++e)
            {
                if (obs_accum_base)
                    std::memset(obs_accum_base + e * obs_spatial_stride, 0, obs_spatial_stride * sizeof(uint8_t));
                if (torch_obs_internal_base)
                    std::memset(torch_obs_internal_base + e * obs_internal_stride, 0, obs_internal_stride * sizeof(float));
                if (torch_state_spatial_base && requires_state)
                    std::memset(torch_state_spatial_base + e * state_spatial_stride, 0, state_spatial_stride * sizeof(float));
                if (torch_state_internal_base && requires_state)
                    std::memset(torch_state_internal_base + e * state_internal_stride, 0, state_internal_stride * sizeof(float));
                if (individual_rewards)
                    std::memset(individual_rewards + e * n_agents, 0, n_agents * sizeof(float));
                if (env_terminated)
                    env_terminated[e] = 0;
                if (env_truncated)
                    env_truncated[e] = 0;
                padded_terminated[e * padded_byte_stride] = 0;
                padded_truncated[e * padded_byte_stride] = 0;
            }
        }

        std::mt19937 base_rng(seed);
        for (int i = 0; i < num_envs; ++i)
        {
            rngs.push_back(std::mt19937(base_rng()));
        }

        if (mode == Mode::DECENTRALIZED)
            decentral_obs_strides = std::make_unique<DecentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
        else if (mode == Mode::CENTRALIZED)
            central_obs_strides = std::make_unique<CentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
        if (torch_state_spatial_base != nullptr && requires_state)
            central_state_strides = std::make_unique<CentralizedStateStrides>(width, height, n_tiles, n_agents);

        arena = std::make_unique<EnvironmentArena>(num_envs, width, height, n_agents, n_pois, n_tiles, mode, init_mode);
        env_views.reserve(num_envs);
        for (int e = 0; e < num_envs; ++e)
            env_views.push_back(arena->get_env_view(e));

        pristine_grid.resize(map_size);
        for (int i = 0; i < map_size; ++i)
        {
            pristine_grid[i].flags = 0;
            pristine_grid[i].altitude = altitude_map[i];

            int t_type = type_map[i];
            pristine_grid[i].set_type(t_type);
            pristine_grid[i].set_walkable(supports_walking[t_type]);
            pristine_grid[i].set_aquatic(supports_aquatic[t_type]);
            pristine_grid[i].set_flyable(supports_flying[t_type]);
            pristine_grid[i].set_blocking(is_blocking[t_type]);
        }

        reset();
        initial_burn_in();
    }

    void initial_burn_in()
    {
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            std::uniform_real_distribution<float> rand_dir(-1.0f, 1.0f);
            std::uniform_int_distribution<int> rand_radio(0, n_agents - 1);

            float max_battery = 0.0f;
            for (int a = 0; a < n_agents; ++a)
            {
                if (env_views[e].agents[a].deployment_remaining > max_battery)
                {
                    max_battery = env_views[e].agents[a].deployment_remaining;
                }
            }

            std::uniform_int_distribution<int> rand_steps(1, std::max(1, static_cast<int>(max_battery)));
            int steps_to_take = rand_steps(rngs[e]);

            std::vector<float> mock_act_data(num_envs * n_agents * 2, 0.0f);
            std::vector<int> mock_radio_data(num_envs * n_agents, 0);

            for (int step = 0; step < steps_to_take; ++step)
            {
                if (padded_terminated[e * padded_byte_stride] || padded_truncated[e * padded_byte_stride])
                    reset_env(e);

                for (int a = 0; a < n_agents; ++a)
                {
                    mock_act_data[e * n_agents * 2 + a * 2] = rand_dir(rngs[e]);
                    mock_act_data[e * n_agents * 2 + a * 2 + 1] = rand_dir(rngs[e]);
                    mock_radio_data[e * n_agents + a] = rand_radio(rngs[e]);
                }

                process_agent_movement(e, mock_act_data.data());
                resolve_local_interactions(e);
                execute_radio(e, mock_radio_data.data());
                update_battery_and_counters(e);

                bool all_saved = (*env_views[e].poi_left == 0);
                bool all_out_of_battery = (*env_views[e].agents_left == 0);
                const bool timeout = current_frames[e] >= max_frames;
                padded_terminated[e * padded_byte_stride] = all_saved || all_out_of_battery;
                padded_truncated[e * padded_byte_stride] = timeout;
                ++current_frames[e];

                if (mode == Mode::DECENTRALIZED || mode == Mode::CENTRALIZED)
                {
                    fill_torch_obs(e);
                    if (ego_view)
                        fill_ego_obs(e);
                }
                if (requires_state)
                    fill_torch_state(e);
            }
        }
    }

    void reset_env(int env_idx)
    {
        current_frames[env_idx] = 0;
        padded_terminated[env_idx * padded_byte_stride] = false;
        padded_truncated[env_idx * padded_byte_stride] = false;

        EnvStateView view = env_views[env_idx];

        if (view.current_frame)
            *view.current_frame = 0;
        if (view.undiscovered_remaining)
            *view.undiscovered_remaining = map_size;

        std::memcpy(view.grid, pristine_grid.data(), map_size * sizeof(Tile));

        for (int a = 0; a < n_agents; ++a)
        {
            view.agents[a].y = init_agent_positions[a * 2];
            view.agents[a].x = init_agent_positions[a * 2 + 1];
            view.agents[a].deployment_remaining = agent_max_batteries[a];

            view.agents[a].flags = agent_flags[a];
            view.agents[a].max_alt_fp16 = float_to_fp16(agent_max_altitudes[a]);
            view.agents[a].last_y = static_cast<uint16_t>(view.agents[a].y);
            view.agents[a].last_x = static_cast<uint16_t>(view.agents[a].x);

            int cy = static_cast<int>(view.agents[a].y);
            int cx = static_cast<int>(view.agents[a].x);

            if (view.agents[a].can_fly())
            {
                float max_alt = fp16_to_float(view.agents[a].max_alt_fp16);
                view.agents[a].view_range = std::max(agent_base_view_ranges[a] * max_alt, 1.0f);
            }
            else
            {
                float start_altitude = view.grid[cy * width + cx].altitude;
                view.agents[a].view_range = std::max(agent_base_view_ranges[a] * start_altitude, 1.0f);
            }

            for (int t = 0; t < n_tiles; ++t)
            {
                view.agent_speeds[a * n_tiles + t] = agent_speed_map[a * n_tiles + t];
            }
        }

        std::uniform_real_distribution<float> dist_w(0.5f, static_cast<float>(width) - 0.5f);
        std::uniform_real_distribution<float> dist_h(0.5f, static_cast<float>(height) - 0.5f);
        for (int p = 0; p < n_pois; ++p)
        {
            view.pois[p].y = dist_h(rngs[env_idx]);
            view.pois[p].x = dist_w(rngs[env_idx]);
            view.pois[p].state = 0;
            view.pois[p].last_y = static_cast<uint16_t>(view.pois[p].y);
            view.pois[p].last_x = static_cast<uint16_t>(view.pois[p].x);

            for (int a = 0; a < n_agents; ++a)
            {
                if (saveable_rules.size() > size_t(p * n_agents + a) && saveable_rules[p * n_agents + a])
                {
                    view.pois[p].set_savable(a, true);
                }
            }
        }

        *view.poi_left = n_pois;
        *view.agents_left = n_agents;

        // Clear the full-map accumulation buffer (fog of war reset). This targets
        // obs_accum_base, which aliases the shared tensor in the non-ego path and
        // the internal buffer in ego mode.
        if (mode == Mode::DECENTRALIZED && obs_accum_base != nullptr)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_size;
            for (int a = 0; a < n_agents; ++a)
            {
                uint8_t *spat_base = obs_accum_base + env_idx * (n_agents * spatial_stride) + a * spatial_stride;
                std::memset(spat_base, 0, spatial_stride * sizeof(uint8_t));
            }
        }
        else if (mode == Mode::CENTRALIZED && obs_accum_base != nullptr)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_size;
            uint8_t *spat_base = obs_accum_base + env_idx * spatial_stride;
            std::memset(spat_base, 0, spatial_stride * sizeof(uint8_t));
        }

        // In ego mode the shared tensor holds the crop, not the accumulation
        // buffer, so zero it directly too. A fresh reset has nothing observed, so
        // an all-zero crop is correct until the next fill_ego_obs.
        if (ego_view && torch_obs_spatial_base != nullptr && mode != Mode::NO_OBS)
        {
            ssize_t ego_env_stride = static_cast<ssize_t>(n_agents) * (n_tiles + 3 + n_agents) * ego_size * ego_size;
            std::memset(torch_obs_spatial_base + env_idx * ego_env_stride, 0, ego_env_stride * sizeof(uint8_t));
        }

        if (requires_state && torch_state_spatial_base != nullptr)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_size;
            float *spat_base = torch_state_spatial_base + env_idx * spatial_stride;
            std::memset(spat_base, 0, spatial_stride * sizeof(float));

            for (int i = 0; i < map_size; ++i)
            {
                spat_base[central_state_strides->TYLE_TYPE_START + view.grid[i].get_type() * map_size + i] = 1.0f;
                spat_base[central_state_strides->ALTITUDE_TYPE_START + i] = view.grid[i].altitude;
            }
        }

        if (mode == Mode::DECENTRALIZED && view.agent_knowledge != nullptr)
        {
            for (int i = 0; i < n_agents * n_agents; ++i)
            {
                view.agent_knowledge[i].has_encountered = 0;
                view.agent_knowledge[i].has_contact = 0;
                view.agent_knowledge[i].last_y = 0;
                view.agent_knowledge[i].last_x = 0;
            }
            for (int i = 0; i < n_agents * n_pois; ++i)
            {
                view.poi_knowledge[i].knows_found = 0;
                view.poi_knowledge[i].knows_saved = 0;
                int p = i % n_pois;
                view.poi_knowledge[i].last_y = static_cast<uint16_t>(view.pois[p].y);
                view.poi_knowledge[i].last_x = static_cast<uint16_t>(view.pois[p].x);
            }
        }
    }

    void reset()
    {
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
            reset_env(e);
    }

    void step(py::array_t<float, py::array::c_style | py::array::forcecast> move_actions_array, py::array_t<int, py::array::c_style> radio_actions_array)
    {
        if ((move_actions_array.ndim() != 1) || (radio_actions_array.ndim() != 1))
            throw std::invalid_argument("actions must have shape [E*A*2] for movement and [E*A] for radio");
        if ((move_actions_array.shape(0) != 2 * num_envs * n_agents) || (radio_actions_array.shape(0) != num_envs * n_agents))
            throw std::invalid_argument("actions first dimension must match num_envs");

        const float *act_data = move_actions_array.data();
        const int *radio_act_data = radio_actions_array.data();

#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            std::memset(padded_rewards.data() + e * padded_reward_stride, 0, n_agents * sizeof(float));

            if (padded_terminated[e * padded_byte_stride] || padded_truncated[e * padded_byte_stride])
                reset_env(e);

            process_agent_movement(e, act_data);
            resolve_local_interactions(e);
            execute_radio(e, radio_act_data);
            update_battery_and_counters(e);

            bool all_saved = (*env_views[e].poi_left == 0);
            bool all_out_of_battery = (*env_views[e].agents_left == 0);
            const bool timeout = current_frames[e] >= max_frames;

            padded_terminated[e * padded_byte_stride] = all_saved || all_out_of_battery;
            padded_truncated[e * padded_byte_stride] = timeout;
            ++current_frames[e];

            if (mode == Mode::DECENTRALIZED || mode == Mode::CENTRALIZED)
            {
                fill_torch_obs(e);
                if (ego_view)
                    fill_ego_obs(e);
            }
            if (requires_state)
                fill_torch_state(e);
        }

        for (int e = 0; e < num_envs; ++e)
        {
            env_terminated[e] = padded_terminated[e * padded_byte_stride];
            env_truncated[e] = padded_truncated[e * padded_byte_stride];
            for (int a = 0; a < n_agents; ++a)
            {
                individual_rewards[e * n_agents + a] = padded_rewards[e * padded_reward_stride + a];
            }
        }

        ++steps_taken;
    }
};

PYBIND11_MODULE(cpp_engine, m)
{
    py::class_<BatchedEnvironment>(m, "BatchedEnvironment")
        .def(py::init<
                 int,
                 int,
                 int,
                 int,
                 std::vector<uint8_t>,
                 std::vector<uint8_t>,
                 std::vector<uint8_t>,
                 std::vector<uint8_t>,
                 std::vector<int>,
                 std::vector<float>,
                 std::vector<float>,
                 std::vector<float>,
                 std::vector<uint8_t>,
                 std::vector<float>,
                 std::vector<float>,
                 std::vector<float>,
                 std::vector<uint8_t>,
                 std::vector<float>,
                 uintptr_t,
                 uintptr_t,
                 uintptr_t,
                 uintptr_t,
                 uintptr_t,
                 uintptr_t,
                 uintptr_t,
                 bool,
                 bool,
                 float,
                 float,
                 float,
                 int,
                 int,
                 int,
                 bool,
                 int>(),
             py::arg("n_envs"),
             py::arg("sim_seed"),
             py::arg("width"),
             py::arg("height"),
             py::arg("supports_walk"),
             py::arg("supports_aqua"),
             py::arg("supports_fly"),
             py::arg("is_block"),
             py::arg("t_map"),
             py::arg("alt_map"),
             py::arg("speed_map"),
             py::arg("agent_view_ranges"),
             py::arg("saveable_rules"),
             py::arg("initial_agent_pos"),
             py::arg("initial_poi_pos"),
             py::arg("agent_max_batteries_val"),
             py::arg("agent_flags_val"),
             py::arg("agent_max_altitudes_val"),
             py::arg("obs_spatial_ptr"),
             py::arg("obs_internal_ptr"),
             py::arg("state_spatial_ptr"),
             py::arg("state_internal_ptr"),
             py::arg("rewards_ptr"),
             py::arg("terminated_ptr"),
             py::arg("truncated_ptr"),
             py::arg("requires_state") = false,
             py::arg("coop_rewards") = true,
             py::arg("reward_new_tile_val") = 0.05f,
             py::arg("reward_found_val") = 2.0f,
             py::arg("reward_saved_val") = 20.0f,
             py::arg("max_frames_val") = 250,
             py::arg("mode_value") = 0,
             py::arg("init_mode") = 0,
             py::arg("ego_view") = false,
             py::arg("ego_size") = 0)
        .def("reset", &BatchedEnvironment::reset)
        .def("reset_env", &BatchedEnvironment::reset_env, py::arg("env_idx"))
        .def("step", &BatchedEnvironment::step, py::arg("move_actions_array"), py::arg("radio_actions_array"));
}