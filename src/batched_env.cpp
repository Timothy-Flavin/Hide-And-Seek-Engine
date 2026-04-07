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
#include "env_state.h"

#include <omp.h>

#ifndef ssize_t
#ifdef _WIN32
typedef ptrdiff_t ssize_t;
#endif
#endif

namespace py = pybind11;

class BatchedEnvironment
{
private:
    std::unique_ptr<EnvironmentArena> arena;
    // Four distinct pinned PyTorch memory spaces
    float *torch_obs_spatial_base = nullptr;
    float *torch_obs_internal_base = nullptr;
    float *torch_state_spatial_base = nullptr;
    float *torch_state_internal_base = nullptr;

    Mode mode;

    // Stride helper objects (conditionally instantiated)
    std::unique_ptr<DecentralizedPartialObsStrides> decentral_obs_strides;
    std::unique_ptr<CentralizedPartialObsStrides> central_obs_strides;
    std::unique_ptr<CentralizedStateStrides> central_state_strides;

    int width;
    int height;
    bool requires_state;
    std::vector<float> individual_rewards;
    // std::vector<float> global_rewards;
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
    unsigned long long steps_taken = 0;

    void process_agent_movement(int e, const float *act_data)
    {
        // for environment e:
        // Takes agent proposed movement direction and if the
        // vector magnitude is over 1.0 it gets reduced to 1.0.
        // Move the agent by its' speed multiplied by the tile speed
        // multiplier of the tile it is standing on and the desired
        // dy,dx vector then handle collisions with blocking tiles
        // by treating the agent as a point collider that needs to
        // get "ejected" to the edge of the tile +0.001 it was trying to enter
        // update the current_tile part of this agent's agent state
        EnvStateView view = env_views[e];

        for (int a = 0; a < n_agents; ++a)
        {
            // 1. Extract and clamp movement vector magnitude
            float dy = act_data[e * n_agents * 2 + a * 2];
            float dx = act_data[e * n_agents * 2 + a * 2 + 1];

            float sqmag = dy * dy + dx * dx;
            if (sqmag > 1.0f)
            {
                float invmag = 1.0f / std::sqrt(sqmag);
                dy *= invmag;
                dx *= invmag;
            }

            AgentState &agent = view.agents[a];

            // Get current integer coordinates safely
            int cy = std::max(0, std::min(height - 1, static_cast<int>(agent.y)));
            int cx = std::max(0, std::min(width - 1, static_cast<int>(agent.x)));

            uint32_t current_tile_type = view.grid[cy * width + cx].get_type();

            // 2. Calculate requested destination
            float speed = view.agent_speeds[a * n_tiles + current_tile_type];
            float ny = agent.y + dy * speed;
            float nx = agent.x + dx * speed;

            // Constrain to map boundaries first
            ny = std::max(0.0f, std::min(static_cast<float>(height) - 0.001f, ny));
            nx = std::max(0.0f, std::min(static_cast<float>(width) - 0.001f, nx));

            // 3. Collision detection against blocking tiles
            int target_y = static_cast<int>(ny);
            int target_x = static_cast<int>(nx);

            Tile &target_tile = view.grid[target_y * width + target_x];

            // If the target tile is blocking, eject the agent to the edge of the tile
            if (target_tile.is_blocking())
            {
                // Simple axis-aligned ejection: determine which axis crossed the boundary
                if (target_y != cy)
                {
                    ny = (dy > 0) ? target_y - 0.001f : target_y + 1.001f;
                }
                if (target_x != cx)
                {
                    nx = (dx > 0) ? target_x - 0.001f : target_x + 1.001f;
                }
            }

            // 4. Update agent state
            agent.y = ny;
            agent.x = nx;

            int final_y = static_cast<int>(ny);
            int final_x = static_cast<int>(nx);
            uint16_t t_id = static_cast<uint16_t>(final_y * width + final_x);
            agent.current_tile = t_id;
            float final_altitude = view.grid[t_id].altitude;
            agent.view_range = std::max(agent_base_view_ranges[a] * final_altitude, 1.0f);
        }
    }

    void resolve_local_interactions(int e)
    {
        // for environment e:
        // Get agent view range from current tile and the tile agent view range
        // For all the POIs, if they are near this agent they become found
        // If this agent is in their rescue list they get rescued
        // global rescued state always set, local information set only if
        // mode decentralized.

        // For all tiles within view range, set their obs for this agent to
        // true and if the global observation is false then accumulate the
        // reward accordingly. Likewise with the agent rescue rewards.
        EnvStateView view = env_views[e];
        const float RESCUE_DIST_SQ = 1.0f * 1.0f; // Must be within 1 unit to rescue

        for (int a = 0; a < n_agents; ++a)
        {
            AgentState &agent = view.agents[a];
            float vr = agent.view_range;
            float vr_sq = vr * vr;

            // 1. Tile Discoveries
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

                        // Global discovery reward
                        if (!t.is_global_observed())
                        {
                            t.set_global_observed();
                            (*view.undiscovered_remaining)--;
                            // Add reward (assume 'rewards' is an accessible class member or passed ref)
                            individual_rewards[e * n_agents + a] += reward_new_tile;
                        }

                        // Local observation tracking
                        if (mode == Mode::DECENTRALIZED)
                        {
                            t.set_agent_seen(a);
                        }
                    }
                }
            }

            // 2. POI (Survivor) Discoveries & Rescues
            for (int p = 0; p < n_pois; ++p)
            {
                POIState &poi = view.pois[p];
                if (poi.saved)
                    continue; // Skip already saved

                float dist_sq = (agent.y - poi.y) * (agent.y - poi.y) + (agent.x - poi.x) * (agent.x - poi.x);

                // If POI is seen
                if (dist_sq <= vr_sq)
                {
                    if (!poi.found)
                    {
                        poi.found = 1;
                        individual_rewards[e * n_agents + a] += reward_found;
                    }

                    // Attempt Rescue
                    if (dist_sq <= RESCUE_DIST_SQ && (poi.savable_by_mask & (1U << a)))
                    {
                        poi.saved = 1;
                        individual_rewards[e * n_agents + a] += reward_saved;
                    }

                    // Update Local Beliefs
                    if (mode == Mode::DECENTRALIZED)
                    {
                        POIKnowledge &pk = view.poi_knowledge[a * n_pois + p];
                        pk.x = poi.x;
                        pk.y = poi.y;
                        pk.knows_found = poi.found;
                        pk.knows_saved = poi.saved;
                    }
                }
            }

            // Maintain self-knowledge
            if (mode == Mode::DECENTRALIZED)
            {
                AgentKnowledge &ak = view.agent_knowledge[a * n_agents + a];
                ak.x = agent.y; // Or agent.x depending on your layout (consistency is key)
                ak.y = agent.x;
                ak.has_contact = 1;
            }
        }
    }

    void execute_radio(int e, const int *radio_act_data)
    {
        // for environment e:
        // If an agent chose radio action 0, it is messaging itself and
        // nothing happens. If it choses another action it is messaging
        // one of the other agents. It will share it's current information
        // updating it's x y in the other agent's tensor observation
        if (mode != Mode::DECENTRALIZED)
            return; // No-op for global and no_obs modes

        EnvStateView view = env_views[e];

        // std::cout << "Starting radio loop " << e << "\n";
        for (int a = 0; a < n_agents; ++a)
        {
            int target_agent = radio_act_data[e * n_agents + a];

            // Assume action maps directly to agent ID (0 = agent 0, 1 = agent 1...)
            // If the environment defines a specific "0 = none, 1 = agent 0" shift, adjust target_agent accordingly.
            if (target_agent == a || target_agent < 0 || target_agent >= n_agents)
                continue;

            // Agent 'a' sends info to 'target_agent'
            // std::cout << "Agent " << a << " sending to " << target_agent << " thread: " << e << "\n";

            // 1. Share 'a' location with 'target_agent'
            AgentKnowledge &target_ak = view.agent_knowledge[target_agent * n_agents + a];
            target_ak.x = view.agents[a].x;
            target_ak.y = view.agents[a].y;
            target_ak.has_contact = 1;

            // std::cout << "share knowledge\n";
            // 2. Share POI knowledge
            for (int p = 0; p < n_pois; ++p)
            {
                POIKnowledge &sender_pk = view.poi_knowledge[a * n_pois + p];
                POIKnowledge &receiver_pk = view.poi_knowledge[target_agent * n_pois + p];

                // If sender knows more, update receiver
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

            // std::cout << "share tile knowledge " << e << "\n";

            // 3. Share Tile Knowledge (Merge maps) ONLY CURRENT VIEW
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
                        if (t.has_agent_seen(a))
                        {
                            t.set_agent_seen(target_agent);
                        }
                    }
                }
            }

            // std::cout << "log " << e << "\n";
        }
    }

    void update_battery_and_counters(int e)
    {
        EnvStateView view = env_views[e];
        for (int a = 0; a < n_agents; ++a)
        {
            // If an agent is past its battery capacity its deployment ends
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
            if (!view.pois[p].saved)
                poi_left++;
        }
        *view.poi_left = poi_left;
    }

    // TODO: slowest possible implementation...
    void fill_torch_obs(int e)
    {
        if (!torch_obs_spatial_base || !torch_obs_internal_base)
            return;

        EnvStateView view = env_views[e];
        ssize_t map_area = width * height;

        if (mode == Mode::DECENTRALIZED)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
            ssize_t internal_stride = 6; // y, x, battery, view_range, deploy, stuck

            for (int a = 0; a < n_agents; ++a)
            {
                float *spat_base = torch_obs_spatial_base + e * (n_agents * spatial_stride) + a * spatial_stride;
                float *int_base = torch_obs_internal_base + e * (n_agents * internal_stride) + a * internal_stride;

                // Spatial MASKED
                float *base_type = spat_base + decentral_obs_strides->TYLE_TYPE_START;
                float *base_alt = spat_base + decentral_obs_strides->ALTITUDE_TYPE_START;
                float *base_obs = spat_base + decentral_obs_strides->OBSERVED_START;

                for (int i = 0; i < map_area; ++i)
                {
                    if (view.grid[i].has_agent_seen(a))
                    {
                        base_type[view.grid[i].get_type() * map_area + i] = 1.0f;
                        base_alt[i] = view.grid[i].altitude;
                        base_obs[i] = 1.0f;
                    }
                }

                // POI MASKED
                for (int p = 0; p < n_pois; ++p)
                {
                    POIKnowledge &pk = view.poi_knowledge[a * n_pois + p];

                    if (pk.last_y >= 0 && pk.last_x >= 0 && pk.last_y < height && pk.last_x < width)
                        spat_base[decentral_obs_strides->PIO_START + pk.last_y * width + pk.last_x] = 0.0f;

                    if (pk.knows_found && !pk.knows_saved)
                    {
                        int py = static_cast<int>(pk.y);
                        int px = static_cast<int>(pk.x);
                        if (py >= 0 && py < height && px >= 0 && px < width)
                        {
                            spat_base[decentral_obs_strides->PIO_START + py * width + px] = 1.0f;
                            pk.last_y = py;
                            pk.last_x = px;
                        }
                        else
                        {
                            pk.last_y = -1;
                            pk.last_x = -1;
                        }
                    }
                    else
                    {
                        pk.last_y = -1;
                        pk.last_x = -1;
                    }
                }

                // Agents
                if (view.agents[a].last_y >= 0 && view.agents[a].last_x >= 0 && view.agents[a].last_y < height && view.agents[a].last_x < width)
                    spat_base[decentral_obs_strides->MY_LOCATION_START + view.agents[a].last_y * width + view.agents[a].last_x] = 0.0f;

                int my_y = static_cast<int>(view.agents[a].y);
                int my_x = static_cast<int>(view.agents[a].x);
                if (my_y >= 0 && my_y < height && my_x >= 0 && my_x < width)
                {
                    spat_base[decentral_obs_strides->MY_LOCATION_START + my_y * width + my_x] = 1.0f;
                    view.agents[a].last_y = my_y;
                    view.agents[a].last_x = my_x;
                }
                else
                {
                    view.agents[a].last_y = -1;
                    view.agents[a].last_x = -1;
                }

                int other_idx = 0;
                for (int a2 = 0; a2 < n_agents; ++a2)
                {
                    if (a == a2)
                        continue;
                    AgentKnowledge &ak = view.agent_knowledge[a * n_agents + a2];

                    if (ak.last_y >= 0 && ak.last_x >= 0 && ak.last_y < height && ak.last_x < width)
                        spat_base[decentral_obs_strides->OTHER_LOCATIONS_START + other_idx * map_area + ak.last_y * width + ak.last_x] = 0.0f;

                    if (ak.has_contact)
                    {
                        int oy = static_cast<int>(ak.y);
                        int ox = static_cast<int>(ak.x);
                        if (oy >= 0 && oy < height && ox >= 0 && ox < width)
                        {
                            spat_base[decentral_obs_strides->OTHER_LOCATIONS_START + other_idx * map_area + oy * width + ox] = 1.0f;
                            ak.last_y = oy;
                            ak.last_x = ox;
                        }
                        else
                        {
                            ak.last_y = -1;
                            ak.last_x = -1;
                        }
                    }
                    else
                    {
                        ak.last_y = -1;
                        ak.last_x = -1;
                    }
                    other_idx++;
                }

                // Internal Vector
                int_base[0] = view.agents[a].y;
                int_base[1] = view.agents[a].x;
                int_base[2] = view.agents[a].battery;
                int_base[3] = view.agents[a].view_range;
                int_base[4] = view.agents[a].deployment_remaining;
                int_base[5] = static_cast<float>(view.agents[a].stuck);
            }
        }
        else if (mode == Mode::CENTRALIZED)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;
            float *spat_base = torch_obs_spatial_base + e * spatial_stride;
            float *int_base = torch_obs_internal_base + e * (n_agents * 6);

            float *base_type = spat_base + central_obs_strides->TYLE_TYPE_START;
            float *base_alt = spat_base + central_obs_strides->ALTITUDE_TYPE_START;
            float *base_obs = spat_base + central_obs_strides->OBSERVED_START;

            for (int i = 0; i < map_area; ++i)
            {
                if (view.grid[i].is_global_observed())
                {
                    base_type[view.grid[i].get_type() * map_area + i] = 1.0f;
                    base_alt[i] = view.grid[i].altitude;
                    base_obs[i] = 1.0f;
                }
            }

            for (int p = 0; p < n_pois; ++p)
            {
                if (view.pois[p].last_y >= 0 && view.pois[p].last_x >= 0 && view.pois[p].last_y < height && view.pois[p].last_x < width)
                    spat_base[central_obs_strides->PIO_START + view.pois[p].last_y * width + view.pois[p].last_x] = 0.0f;

                if (view.pois[p].found && !view.pois[p].saved)
                {
                    int py = static_cast<int>(view.pois[p].y);
                    int px = static_cast<int>(view.pois[p].x);
                    if (py >= 0 && py < height && px >= 0 && px < width)
                    {
                        spat_base[central_obs_strides->PIO_START + py * width + px] = 1.0f;
                        view.pois[p].last_y = py;
                        view.pois[p].last_x = px;
                    }
                    else
                    {
                        view.pois[p].last_y = -1;
                        view.pois[p].last_x = -1;
                    }
                }
                else
                {
                    view.pois[p].last_y = -1;
                    view.pois[p].last_x = -1;
                }
            }

            for (int a = 0; a < n_agents; ++a)
            {
                if (view.agents[a].last_y >= 0 && view.agents[a].last_x >= 0 && view.agents[a].last_y < height && view.agents[a].last_x < width)
                    spat_base[central_obs_strides->AGENT_LOCATIONS_START + a * map_area + view.agents[a].last_y * width + view.agents[a].last_x] = 0.0f;

                int ay = static_cast<int>(view.agents[a].y);
                int ax = static_cast<int>(view.agents[a].x);
                if (ay >= 0 && ay < height && ax >= 0 && ax < width)
                {
                    spat_base[central_obs_strides->AGENT_LOCATIONS_START + a * map_area + ay * width + ax] = 1.0f;
                    view.agents[a].last_y = ay;
                    view.agents[a].last_x = ax;
                }
                else
                {
                    view.agents[a].last_y = -1;
                    view.agents[a].last_x = -1;
                }

                int_base[a * 6 + 0] = view.agents[a].y;
                int_base[a * 6 + 1] = view.agents[a].x;
                int_base[a * 6 + 2] = view.agents[a].battery;
                int_base[a * 6 + 3] = view.agents[a].view_range;
                int_base[a * 6 + 4] = view.agents[a].deployment_remaining;
                int_base[a * 6 + 5] = static_cast<float>(view.agents[a].stuck);
            }
        }
    }
    // TODO: slowest possible implementation...
    void fill_torch_state(int e)
    {
        if (!torch_state_spatial_base || !torch_state_internal_base)
            return;

        EnvStateView view = env_views[e];
        ssize_t map_area = width * height;
        ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_area;

        float *spat_base = torch_state_spatial_base + e * spatial_stride;
        float *int_base = torch_state_internal_base + e * (n_agents * 6 + n_pois * 4);

        // UNMASKED Terrain
        for (int i = 0; i < map_area; ++i)
        {
            float altitude = view.grid[i].altitude;
            spat_base[central_state_strides->TYLE_TYPE_START + view.grid[i].get_type() * map_area + i] = 1.0f;
            spat_base[central_state_strides->ALTITUDE_TYPE_START + i] = altitude;
            if (view.grid[i].is_global_observed())
            {
                spat_base[central_state_strides->OBSERVED_START + i] = 1.0f;
            }
        }

        // UNMASKED POIs (Any POI not yet saved)
        for (int p = 0; p < n_pois; ++p)
        {
            if (!view.pois[p].saved)
            {
                int py = static_cast<int>(view.pois[p].y);
                int px = static_cast<int>(view.pois[p].x);
                if (py >= 0 && py < height && px >= 0 && px < width)
                {
                    spat_base[central_state_strides->PIO_START + py * width + px] = 1.0f;
                }
            }
        }

        // True Agent Locations
        int int_off = 0;
        for (int a = 0; a < n_agents; ++a)
        {
            int ay = static_cast<int>(view.agents[a].y);
            int ax = static_cast<int>(view.agents[a].x);
            if (ay >= 0 && ay < height && ax >= 0 && ax < width)
            {
                spat_base[central_state_strides->AGENT_LOCATIONS_START + a * map_area + ay * width + ax] = 1.0f;
            }
            int_base[int_off++] = view.agents[a].y;
            int_base[int_off++] = view.agents[a].x;
            int_base[int_off++] = view.agents[a].battery;
            int_base[int_off++] = view.agents[a].view_range;
            int_base[int_off++] = view.agents[a].deployment_remaining;
            int_base[int_off++] = static_cast<float>(view.agents[a].stuck);
        }

        // True POI data
        for (int p = 0; p < n_pois; ++p)
        {
            int_base[int_off++] = view.pois[p].y;
            int_base[int_off++] = view.pois[p].x;
            int_base[int_off++] = static_cast<float>(view.pois[p].found);
            int_base[int_off++] = static_cast<float>(view.pois[p].saved);
        }
    }

public:
    // Hyper Parameters sent from python
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

    // Internal data for resetting the environment managing randomness
    // and rendering the radio
    std::vector<std::mt19937> rngs;
    std::vector<bool> env_terminated;
    std::vector<bool> env_truncated;
    std::vector<int> current_frames;
    std::vector<float> init_agent_positions;
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
        const std::vector<int> &t_map, // width*height int for tile types
        const std::vector<float> &alt_map,
        const std::vector<float> &speed_map,
        const std::vector<float> &agent_view_ranges,
        const std::vector<uint8_t> &saveable_rules,  // n_poi * n_agents can this poi be saved by this agent
        const std::vector<float> &initial_agent_pos, // n_agent * 2 x y start locs
        const std::vector<float> &initial_poi_pos,   // n_poi * 2 x y start locs
        uintptr_t obs_spatial_ptr,
        uintptr_t obs_internal_ptr,
        uintptr_t state_spatial_ptr,
        uintptr_t state_internal_ptr,
        bool requires_state = false, // Should we allocate the true state tensor?
        bool coop_rewards = true,
        float reward_new_tile_val = 0.05f,
        float reward_found_val = 2.0f,
        float reward_saved_val = 20.0f,
        int max_frames_val = 250,
        int mode_value = 0)
        : num_envs(n_envs), seed(sim_seed), width(w), height(h),
          supports_walking(std::move(supports_walk)), supports_aquatic(std::move(supports_aqua)),
          supports_flying(std::move(supports_fly)), is_blocking(std::move(is_block)),
          type_map(std::move(t_map)), altitude_map(std::move(alt_map)), agent_speed_map(std::move(speed_map)),
          agent_base_view_ranges(std::move(agent_view_ranges)),
          saveable_rules(std::move(saveable_rules)), init_agent_positions(std::move(initial_agent_pos)),
          init_poi_positions(std::move(initial_poi_pos)), requires_state(requires_state), cooperative_rewards(coop_rewards),
          reward_new_tile(reward_new_tile_val), reward_found(reward_found_val), reward_saved(reward_saved_val),
          max_frames(max_frames_val), map_size(w * h), mode(static_cast<Mode>(mode_value))
    {
        if (mode_value < 0 || mode_value > 2)
        {
            throw(std::invalid_argument("mode value < 0 or > 2 cannot be bound into Mode enum"));
        }
        // 1. Dynamic bounds tracking
        n_tiles = int(supports_walking.size());
        n_agents = int(init_agent_positions.size() / 2);
        n_pois = int(init_poi_positions.size() / 2);

        // 2. Setup internal simulation states
        env_terminated.resize(num_envs, false);
        env_truncated.resize(num_envs, false);
        current_frames.resize(num_envs, 0);
        radio_logs.resize(num_envs, "");
        individual_rewards.resize(num_envs * n_agents, 0.0f);

        torch_obs_spatial_base = reinterpret_cast<float *>(obs_spatial_ptr);
        torch_obs_internal_base = reinterpret_cast<float *>(obs_internal_ptr);
        torch_state_spatial_base = reinterpret_cast<float *>(state_spatial_ptr);
        torch_state_internal_base = reinterpret_cast<float *>(state_internal_ptr);

        std::mt19937 base_rng(seed);
        for (int i = 0; i < num_envs; ++i)
        {
            rngs.push_back(std::mt19937(base_rng()));
        }

        // Bind Stride Structs purely from env_state.h truth
        if (mode == Mode::DECENTRALIZED)
        {
            decentral_obs_strides = std::make_unique<DecentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
        }
        else if (mode == Mode::CENTRALIZED)
        {
            central_obs_strides = std::make_unique<CentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
        }
        if (torch_state_spatial_base != nullptr && requires_state)
        {
            central_state_strides = std::make_unique<CentralizedStateStrides>(width, height, n_tiles, n_agents);
        }

        arena = std::make_unique<EnvironmentArena>(num_envs, width, height, n_agents, n_pois, n_tiles, mode);
        env_views.reserve(num_envs);
        for (int e = 0; e < num_envs; ++e)
            env_views.push_back(arena->get_env_view(e));

        // Create the base grid once to copy during resets
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

            // Find max battery among agents for this environment
            float max_battery = 0.0f;
            for (int a = 0; a < n_agents; ++a)
            {
                if (env_views[e].agents[a].battery > max_battery)
                {
                    max_battery = env_views[e].agents[a].battery;
                }
            }

            std::uniform_int_distribution<int> rand_steps(1, std::max(1, static_cast<int>(max_battery)));
            int steps_to_take = rand_steps(rngs[e]);

            std::vector<float> mock_act_data(num_envs * n_agents * 2, 0.0f);
            std::vector<int> mock_radio_data(num_envs * n_agents, 0);

            for (int step = 0; step < steps_to_take; ++step)
            {
                if (env_terminated[e] || env_truncated[e])
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
                env_terminated[e] = all_saved || all_out_of_battery;
                env_truncated[e] = timeout;
                ++current_frames[e];

                if (mode == Mode::DECENTRALIZED || mode == Mode::CENTRALIZED)
                {
                    fill_torch_obs(e);
                }
                if (requires_state)
                {
                    fill_torch_state(e);
                }
            }
        }
    }

    void reset_env(int env_idx)
    {
        current_frames[env_idx] = 0;
        env_terminated[env_idx] = false;
        env_truncated[env_idx] = false;

        EnvStateView view = env_views[env_idx];

        // A. Env Counters
        if (view.current_frame)
            *view.current_frame = 0;
        if (view.undiscovered_remaining)
            *view.undiscovered_remaining = map_size;

        // B. Tiles
        std::memcpy(view.grid, pristine_grid.data(), map_size * sizeof(Tile));

        // C. Agents
        for (int a = 0; a < n_agents; ++a)
        {
            view.agents[a].y = init_agent_positions[a * 2];
            view.agents[a].x = init_agent_positions[a * 2 + 1];
            view.agents[a].battery = 100.0f;

            int cy = static_cast<int>(view.agents[a].y);
            int cx = static_cast<int>(view.agents[a].x);
            float start_altitude = view.grid[cy * width + cx].altitude;
            view.agents[a].view_range = std::max(agent_base_view_ranges[a] * start_altitude, 1.0f);

            view.agents[a].deployment_remaining = view.agents[a].battery;
            view.agents[a].type = a;
            view.agents[a].stuck = 0;

            // Load specialized speeds
            for (int t = 0; t < n_tiles; ++t)
            {
                view.agent_speeds[a * n_agents + t] = agent_speed_map[a * n_tiles + t];
            }
        }

        // D. Survivors (POIs)
        std::uniform_real_distribution<float> dist_w(0.5f, static_cast<float>(width) - 0.5f);
        std::uniform_real_distribution<float> dist_h(0.5f, static_cast<float>(height) - 0.5f);
        for (int p = 0; p < n_pois; ++p)
        {
            view.pois[p].y = dist_h(rngs[env_idx]);
            view.pois[p].x = dist_w(rngs[env_idx]);
            view.pois[p].found = 0;
            view.pois[p].saved = 0;
            view.pois[p].moves = 0;

            // Generate 1-hot bitmask integer evaluating the allowed savers array natively
            uint32_t savable_mask = 0;
            for (int a = 0; a < n_agents; ++a)
            {
                if (saveable_rules.size() > size_t(p * n_agents + a) && saveable_rules[p * n_agents + a])
                {
                    savable_mask |= (1U << a);
                }
            }
            view.pois[p].savable_by_mask = savable_mask;
        }

        *view.poi_left = n_pois;
        *view.agents_left = n_agents;

        // E. Diff Tracker & Tensor Clear
        if (mode == Mode::DECENTRALIZED && torch_obs_spatial_base != nullptr)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_size;
            for (int a = 0; a < n_agents; ++a)
            {
                float *spat_base = torch_obs_spatial_base + env_idx * (n_agents * spatial_stride) + a * spatial_stride;
                std::memset(spat_base, 0, spatial_stride * sizeof(float));
            }
        }
        else if (mode == Mode::CENTRALIZED && torch_obs_spatial_base != nullptr)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_size;
            float *spat_base = torch_obs_spatial_base + env_idx * spatial_stride;
            std::memset(spat_base, 0, spatial_stride * sizeof(float));
        }

        if (requires_state && torch_state_spatial_base != nullptr)
        {
            ssize_t spatial_stride = (n_tiles + 3 + n_agents) * map_size;
            float *spat_base = torch_state_spatial_base + env_idx * spatial_stride;
            std::memset(spat_base, 0, spatial_stride * sizeof(float));
        }

        for (int a = 0; a < n_agents; ++a)
        {
            view.agents[a].last_x = -1;
            view.agents[a].last_y = -1;
            view.agents[a].state_last_x = -1;
            view.agents[a].state_last_y = -1;
        }
        for (int p = 0; p < n_pois; ++p)
        {
            view.pois[p].last_x = -1;
            view.pois[p].last_y = -1;
            view.pois[p].state_last_x = -1;
            view.pois[p].state_last_y = -1;
        }

        if (mode == Mode::DECENTRALIZED && view.agent_knowledge != nullptr)
        {
            for (int i = 0; i < n_agents * n_agents; ++i)
            {
                view.agent_knowledge[i].last_x = -1;
                view.agent_knowledge[i].last_y = -1;
            }
            for (int i = 0; i < n_agents * n_pois; ++i)
            {
                view.poi_knowledge[i].last_x = -1;
                view.poi_knowledge[i].last_y = -1;
            }
        }
    }

    void reset()
    {
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
            reset_env(e);
    }

    // Timing accumulators
    inline static std::chrono::duration<double> t_process_movement{0};
    inline static std::chrono::duration<double> t_resolve_interactions{0};
    inline static std::chrono::duration<double> t_radio_logs{0};
    inline static std::chrono::duration<double> t_update_counters{0};
    inline static std::chrono::duration<double> t_post_processing{0};
    inline static std::chrono::duration<double> t_fill_torch_obs{0};
    inline static std::chrono::duration<double> t_fill_torch_state{0};
    inline static std::chrono::duration<double> t_reset_time{0};
    inline static std::chrono::duration<double> t_python_overhead{0};
    inline static std::chrono::duration<double> t_total{0};

    py::tuple step(py::array_t<float, py::array::c_style | py::array::forcecast> move_actions_array, py::array_t<int, py::array::c_style> radio_actions_array)
    {
        auto t_step_start = std::chrono::high_resolution_clock::now();
        // action space is box2d[dx, dy], discrete(num radio choices)
        if ((move_actions_array.ndim() != 1) || (radio_actions_array.ndim() != 1))
            throw std::invalid_argument("actions must have shape [E*A*2] for movement and [E*A] for radio");
        if ((move_actions_array.shape(0) != 2 * num_envs * n_agents) || (radio_actions_array.shape(0) != num_envs * n_agents))
            throw std::invalid_argument("actions first dimension must match num_envs");

        const float *act_data = move_actions_array.data();
        const int *radio_act_data = radio_actions_array.data();

        std::fill(individual_rewards.begin(), individual_rewards.end(), 0.0f);

        auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            auto t_reset_s = std::chrono::high_resolution_clock::now();
            if (env_terminated[e] || env_truncated[e])
                reset_env(e);

            // std::cout << "Env " << e << " process movement\n";
            auto t1 = std::chrono::high_resolution_clock::now();
            process_agent_movement(e, act_data);
            auto t2 = std::chrono::high_resolution_clock::now();
            // std::cout << "Env " << e << " resolve interactions\n";
            resolve_local_interactions(e);
            auto t3 = std::chrono::high_resolution_clock::now();
            // std::cout << "Env " << e << " radio logs\n";
            execute_radio(e, radio_act_data);
            auto t4 = std::chrono::high_resolution_clock::now();
            // std::cout << "Env " << e << " update counters\n";
            update_battery_and_counters(e);
            auto t5 = std::chrono::high_resolution_clock::now();

            // std::cout << "Env " << e << " post processing\n";
            bool all_saved = (*env_views[e].poi_left == 0);
            bool all_out_of_battery = (*env_views[e].agents_left == 0);
            const bool timeout = current_frames[e] >= max_frames;
            env_terminated[e] = all_saved || all_out_of_battery;
            env_truncated[e] = timeout;
            ++current_frames[e];
            auto t6 = std::chrono::high_resolution_clock::now();

            // If we are using this environment for machine learning
            // then fill the torch buffers accordingly
            // std::cout << "Env " << e << " fill torch\n";
            if (mode == Mode::DECENTRALIZED || mode == Mode::CENTRALIZED)
            {
                fill_torch_obs(e);
            }
            auto t7 = std::chrono::high_resolution_clock::now();
            if (requires_state)
            {
                fill_torch_state(e);
            }
            auto t8 = std::chrono::high_resolution_clock::now();

#pragma omp critical
            {
                t_reset_time += (t1 - t_reset_s);
                t_process_movement += (t2 - t1);
                t_resolve_interactions += (t3 - t2);
                t_radio_logs += (t4 - t3);
                t_update_counters += (t5 - t4);
                t_post_processing += (t6 - t5);
                t_fill_torch_obs += (t7 - t6);
                t_fill_torch_state += (t8 - t7);
            }
        }
        auto t9 = std::chrono::high_resolution_clock::now();

        // std::cout << "setup for python" << std::endl;
        auto py_rewards = py::array_t<float>({num_envs, n_agents});
        auto py_terminated = py::array_t<bool>(num_envs);
        auto py_truncated = py::array_t<bool>(num_envs);
        std::copy(individual_rewards.begin(), individual_rewards.end(), py_rewards.mutable_data());
        std::copy(env_terminated.begin(), env_terminated.end(), py_terminated.mutable_data());
        std::copy(env_truncated.begin(), env_truncated.end(), py_truncated.mutable_data());
        auto pytup = py::make_tuple(py_rewards, py_terminated, py_truncated);
        auto t_step_end = std::chrono::high_resolution_clock::now();
        t_total += (t_step_end - t_step_start);
        t_python_overhead += (t0 - t_step_start) + (t_step_end - t9);

        ++steps_taken;
        if (steps_taken % 10000 == 0)
        {
            std::cout << "[Timing after " << steps_taken << " steps]" << std::endl;
            std::cout << "  process_agent_movement: " << t_process_movement.count() << " s" << std::endl;
            std::cout << "  resolve_local_interactions: " << t_resolve_interactions.count() << " s" << std::endl;
            std::cout << "  execute_radio: " << t_radio_logs.count() << " s" << std::endl;
            std::cout << "  update_battery_and_counters: " << t_update_counters.count() << " s" << std::endl;
            std::cout << "  post_processing: " << t_post_processing.count() << " s" << std::endl;
            std::cout << "  fill_torch_obs: " << t_fill_torch_obs.count() << " s" << std::endl;
            std::cout << "  fill_torch_state: " << t_fill_torch_state.count() << " s" << std::endl;
            std::cout << "  reset_env: " << t_reset_time.count() << " s" << std::endl;
            std::cout << "  python_overhead: " << t_python_overhead.count() << " s" << std::endl;
            std::cout << "  TOTAL step() time: " << t_total.count() << " s" << std::endl;
            std::cout << std::endl;
        }
        // Optionally reset timers if you want per-interval stats
        //     t_process_movement = std::chrono::duration<double>(0);
        //     t_resolve_interactions = std::chrono::duration<double>(0);
        //     t_radio_logs = std::chrono::duration<double>(0);
        //     t_update_counters = std::chrono::duration<double>(0);
        //     t_post_processing = std::chrono::duration<double>(0);
        //     t_fill_torch_obs = std::chrono::duration<double>(0);
        //     t_fill_torch_state = std::chrono::duration<double>(0);
        //     t_reset_time = std::chrono::duration<double>(0);
        //     t_python_overhead = std::chrono::duration<double>(0);
        //     t_total = std::chrono::duration<double>(0);
        // }
        
        return pytup;
    }
};

PYBIND11_MODULE(cpp_engine, m)
{
    /* REGISTER_FEATURE_TYPE_ENUM(m); // Uncomment or define if needed */

    py::class_<BatchedEnvironment>(m, "BatchedEnvironment")
        .def(py::init<
                 int,                  // n_envs
                 int,                  // sim_seed
                 int,                  // width
                 int,                  // height
                 std::vector<uint8_t>, // supports_walk
                 std::vector<uint8_t>, // supports_aqua
                 std::vector<uint8_t>, // supports_fly
                 std::vector<uint8_t>, // is_block
                 std::vector<int>,     // t_map
                 std::vector<float>,   // alt_map
                 std::vector<float>,   // speed_map
                 std::vector<float>,   // agent_view_ranges
                 std::vector<uint8_t>, // saveable_rules
                 std::vector<float>,   // initial_agent_pos
                 std::vector<float>,   // initial_poi_pos
                 uintptr_t,            // obs_spatial_ptr
                 uintptr_t,            // obs_internal_ptr
                 uintptr_t,            // state_spatial_ptr
                 uintptr_t,            // state_internal_ptr
                 bool,                 // requires_state
                 bool,                 // coop_rewards
                 float,                // reward_new_tile_val
                 float,                // reward_found_val
                 float,                // reward_saved_val
                 int,                  // max_frames_val
                 int                   // mode_value
                 >(),
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
             py::arg("obs_spatial_ptr"),
             py::arg("obs_internal_ptr"),
             py::arg("state_spatial_ptr"),
             py::arg("state_internal_ptr"),
             py::arg("requires_state") = false,
             py::arg("coop_rewards") = true,
             py::arg("reward_new_tile_val") = 0.05f,
             py::arg("reward_found_val") = 2.0f,
             py::arg("reward_saved_val") = 20.0f,
             py::arg("max_frames_val") = 250,
             py::arg("mode_value") = 0)
        .def("reset", &BatchedEnvironment::reset)
        .def("reset_env", &BatchedEnvironment::reset_env, py::arg("env_idx"))
        .def("step", &BatchedEnvironment::step, py::arg("move_actions_array"), py::arg("radio_actions_array"))
        // Add other methods as needed
        ;
}
