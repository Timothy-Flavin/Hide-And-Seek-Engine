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
#inlcude "env_state.h"

#include <omp.h>

#include "gravity.h"

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
    float *torch_tensor_base = nullptr;
    float *torch_state_tensor_base = nullptr;

    // Stride helper objects (conditionally instantiated)
    std::unique_ptr<DecentralizedPartialObsStrides> decentral_obs_strides;
    std::unique_ptr<CentralizedPartialObsStrides> central_obs_strides;
    std::unique_ptr<CentralizedStateStrides> central_state_strides;

    // Persisted map configuration for reset_env
    int width;
    int height;
    std::vector<bool> supports_walking;
    std::vector<bool> supports_aquatic;
    std::vector<bool> supports_flying;
    std::vector<bool> is_blocking;
    std::vector<int> type_map;
    std::vector<float> altitude_map;
    std::vector<float> agent_speed_map;
    std::vector<bool> saveable_map;

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
    int max_frames;
    int num_envs;
    int map_size;
    bool return_obs = False;
    // Internal data for resetting the environment managing randomness
    // and rendering the radio
    std::vector<std::mt19937> rngs;
    std::vector<bool> env_terminated;
    std::vector<int> current_frames;
    std::vector<float> init_agent_positions;
    std::vector<float> init_poi_positions;
    std::vector<std::string> radio_logs;

    BatchedEnvironment(
        int n_envs,
        int sim_seed,
        int w,
        int h,
        std::vector<bool> supports_walk,
        std::vector<bool> supports_aqua,
        std::vector<bool> supports_fly,
        std::vector<bool> is_block,
        std::vector<int> t_map,               // width*height int for tile types
        std::vector<float> alt_map,           // width*height altitudes
        std::vector<float> a_speed_map,       // n_agents * n_tile_types speed of each agent on each tile
        std::vector<bool> save_map,           // n_poi * n_agents can this poi be saved by this agent
        std::vector<float> initial_agent_pos, // n_agent * 2 x y start locs
        std::vector<float> initial_poi_pos,   // n_poi * 2 x y start locs
        uintptr_t tensor_ptr,                 // PyTorch pinned tensor pointer mapped via data_ptr() (Observations)
        std::string mode = "decentralized",   // "centralized" or "decentralized"
        bool requires_state = false,          // Should we allocate the true state tensor?
        uintptr_t state_tensor_ptr = 0,       // PyTorch pinned tensor pointer mapped via data_ptr() (Global State)
        bool coop_rewards = true,
        float reward_new_tile_val = 0.05f,
        float reward_found_val = 2.0f,
        float reward_saved_val = 20.0f,
        int max_frames_val = 250)
        : num_envs(n_envs),
          seed(sim_seed),
          width(w),
          height(h),
          supports_walking(std::move(supports_walk)),
          supports_aquatic(std::move(supports_aqua)),
          supports_flying(std::move(supports_fly)),
          is_blocking(std::move(is_block)),
          type_map(std::move(t_map)),
          altitude_map(std::move(alt_map)),
          agent_speed_map(std::move(a_speed_map)),
          saveable_map(std::move(save_map)),
          init_agent_positions(std::move(initial_agent_pos)),
          init_poi_positions(std::move(initial_poi_pos)),
          cooperative_rewards(coop_rewards),
          reward_new_tile(reward_new_tile_val),
          reward_found(reward_found_val),
          reward_saved(reward_saved_val),
          max_frames(max_frames_val),
          map_size(w * h)
    {
        // 1. Dynamic bounds tracking
        n_tiles = supports_walking.size();
        n_agents = init_agent_positions.size() / 2;
        n_pois = init_poi_positions.size() / 2;

        // 2. Setup internal simulation states
        env_terminated.resize(num_envs, false);
        current_frames.resize(num_envs, 0);
        radio_logs.resize(num_envs, "");

        torch_tensor_base = reinterpret_cast<float *>(tensor_ptr);

        std::mt19937 base_rng(seed);
        for (int i = 0; i < num_envs; ++i)
        {
            rngs.push_back(std::mt19937(base_rng()));
        }

        // 3. Allocate cache-aligned contiguous block memory
        arena = std::make_unique<EnvironmentArena>(num_envs, width, height, n_agents, n_pois, n_tiles);

        // 4. Bind the required tensor stride objects based on configuration
        bind_state(mode, requires_state, state_tensor_ptr);

        // 5. Fill the initial game state structure
        for (int env_idx = 0; env_idx < num_envs; ++env_idx)
        {
            reset_env(env_idx);
        }
    }

    void bind_state(const std::string &mode, bool requires_state, uintptr_t state_tensor_ptr)
    {
        if (mode == "centralized")
        {
            central_obs_strides = std::make_unique<CentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
        }
        else // default to decentralized
        {
            decentral_obs_strides = std::make_unique<DecentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
        }

        if (requires_state)
        {
            central_state_strides = std::make_unique<CentralizedStateStrides>(width, height, n_tiles, n_agents);
            torch_state_tensor_base = reinterpret_cast<float *>(state_tensor_ptr);
        }
    }

    void reset_env(int env_idx)
    {
        EnvStateView view = arena->get_env_view(env_idx);

        // A. Env Counters
        *view.current_frame = 0;
        *view.undiscovered_remaining = map_size;

        // B. Tiles
        for (int i = 0; i < map_size; ++i)
        {
            view.grid[i].flags = 0; // zero out completely
            view.grid[i].altitude = altitude_map[i];

            int t_type = type_map[i];
            view.grid[i].set_type(t_type);
            view.grid[i].set_walkable(supports_walking[t_type]);
            view.grid[i].set_aquatic(supports_aquatic[t_type]);
            view.grid[i].set_flyable(supports_flying[t_type]);
            view.grid[i].set_blocking(is_blocking[t_type]);
        }

        // C. Agents
        for (int a = 0; a < n_agents; ++a)
        {
            view.agents[a].y = init_agent_positions[a * 2];
            view.agents[a].x = init_agent_positions[a * 2 + 1];
            view.agents[a].battery = 100.0f;
            view.agents[a].view_range = 5.0f;
            view.agents[a].deployment_remaining = 0.0f;
            view.agents[a].type = a;
            view.agents[a].stuck = 0;

            // Load specialized speeds
            for (int t = 0; t < n_tiles; ++t)
            {
                view.agent_speeds[a * n_tiles + t] = agent_speed_map[a * n_tiles + t];
            }
        }

        // D. Survivors (POIs)
        for (int p = 0; p < n_pois; ++p)
        {
            view.pois[p].y = init_poi_positions[p * 2];
            view.pois[p].x = init_poi_positions[p * 2 + 1];
            view.pois[p].found = 0;
            view.pois[p].saved = 0;
            view.pois[p].moves = 0;

            // Generate 1-hot bitmask integer evaluating the allowed savers array natively
            uint32_t savable_mask = 0;
            for (int a = 0; a < n_agents; ++a)
            {
                if (saveable_map[p * n_agents + a])
                {
                    savable_mask |= (1U << a);
                }
            }
            view.pois[p].savable_by_mask = savable_mask;
        }
    }

    void reset()
    {
        env_terminated.assign(num_envs, false);
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            reset_env(e);
        }
    }

    void step(py::array_t<float, py::array::c_style | py::array::forcecast> move_actions_array, py::array_t<int, py::array::c_style> radio_actions_array)
    {
        // action space is box2d[dx, dy], discrete(num radio choices)
        if (actions_array.ndim() != 2 && actions_array.ndim() != 3)
            throw std::invalid_argument("actions must have shape [E, A*3] or [E, A, 3]");
        if (actions_array.shape(0) != num_envs)
            throw std::invalid_argument("actions first dimension must match num_envs");

        const float *act_data = actions_array.data();
        const int *radio_act_data = radio_actions_array.data();

#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {

            for (int i = 0; i < N_AGENTS; ++i)
                rewards(e, i) = 0.0f;

            if (env_terminated[e])
            {
                reset_env(e); // for parallel env reset is automatic
            }

            // figure out where an agent will end up this frame
            // and set its current location and grid cell accordingly
            process_agent_movement(e, act_data);
            // Resolve tile discoveries and POI discoveries / rescues
            resolve_local_interactions(e);
            // Resolve or at least record which agents have shared
            // their internal states with other agents
            radio_logs[e] = execute_radio(s, radio_act_data);

            const bool all_saved = all_pois_saved(s);
            const bool timeout = current_frames[e] >= max_frames;
            terminated(e) = all_saved || timeout;
            env_terminated[e] = terminated(e);

            // If we are using this environment for machine learning
            // then fill the torch buffer accordingly
            if (return_obs)
                fill_torch_state(e);
        }

        return {rewards, terminated, truncated};
    }

private:
    void process_agent_movement(int e, float *act_data)
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
    }
    void resolve_local_interactions(int e)
    {
        // for environment e:
        // Get agent view range from current tile and the tile agent view range
        // For all the POIs, if they are near this agent they become found
        // If this agent is in their rescue list they get rescued

        // For all tiles within view range, set their obs for this agent to
        // true and if the global observation is false then accumulate the
        // reward accordingly. Likewise with the agent rescue rewards.
    }
    execute_radio(int e, const int *radio_act_data)
    {
        // for environment e:
        // If an agent chose radio action 0, it is messaging itself and
        // nothing happens. If it choses another action it is messaging
        // one of the other agents. It will share it's current information
        // updating it's x y in the other agent's tensor observation for
        // other
    }

    MoveEval evaluate_move(const GameStateView &s, int env_idx, int agent_idx, int dy, int dx) const
    {
        const int cy = std::clamp(static_cast<int>(s.agent_positions[agent_idx * 2]), 0, MAP_SIZE - 1);
        const int cx = std::clamp(static_cast<int>(s.agent_positions[agent_idx * 2 + 1]), 0, MAP_SIZE - 1);
        const int ny = cy + dy;
        const int nx = cx + dx;
        if (ny < 0 || ny >= MAP_SIZE || nx < 0 || nx >= MAP_SIZE)
            return MoveEval::BLOCKED;

        const int tile_id = terrain_id_at(env_idx, ny, nx);
        const bool can_fly = agent_spec(agent_idx, 0) > 0.5f;
        const bool can_aquatic = agent_spec(agent_idx, 1) > 0.5f;
        const bool can_walk = agent_spec(agent_idx, 2) > 0.5f;

        const bool tile_walk = tile_supports_walking[tile_id] > 0;
        const bool tile_aq = tile_supports_aquatic[tile_id] > 0;
        const bool tile_fl = tile_supports_flying[tile_id] > 0;
        const bool has_supported_mode =
            (tile_walk && can_walk) ||
            (tile_aq && can_aquatic) ||
            (tile_fl && can_fly);

        const float alt = altitude_at(s, ny, nx);
        const float alt_min = agent_spec(agent_idx, 3);
        const float alt_max = agent_spec(agent_idx, 4);
        if (alt < alt_min || alt > alt_max)
            return MoveEval::BLOCKED;

        if (has_supported_mode)
            return MoveEval::ALLOW;

        return tile_blocking[tile_id] > 0 ? MoveEval::BLOCKED : MoveEval::ALLOW_AND_STUCK;
    }

    bool is_move_legal(const GameStateView &s, int env_idx, int agent_idx, int dy, int dx) const
    {
        return evaluate_move(s, env_idx, agent_idx, dy, dx) != MoveEval::BLOCKED;
    }

    void process_agent_movement(
        GameStateView &s,
        int env_idx,
        const float *act_data,
        int action_ndim,
        py::detail::unchecked_mutable_reference<float, 2> &rewards,
        std::array<int, N_AGENTS> &radio_actions)
    {
        for (int i = 0; i < N_AGENTS; ++i)
        {
            if (s.agent_deployment_remaining[i] > 0.0f)
            {
                s.agent_deployment_remaining[i] = std::max(0.0f, s.agent_deployment_remaining[i] - 1.0f);
                continue;
            }
            if (s.agent_stuck[i] > 0.5f)
                continue;

            const size_t base = action_ndim == 2
                                    ? static_cast<size_t>(env_idx) * (N_AGENTS * 3) + i * 3
                                    : static_cast<size_t>(env_idx) * N_AGENTS * 3 + i * 3;
            float dy = act_data[base];
            float dx = act_data[base + 1];
            const float radio = act_data[base + 2];

            radio_actions[i] = std::clamp(static_cast<int>(std::round(radio)), 0, 3);

            const float len_sq = dy * dy + dx * dx;
            if (len_sq > 1e-8f)
            {
                const float inv_len = 1.0f / std::sqrt(len_sq);
                dy *= inv_len;
                dx *= inv_len;
            }

            const int cy = std::clamp(static_cast<int>(s.agent_positions[i * 2]), 0, MAP_SIZE - 1);
            const int cx = std::clamp(static_cast<int>(s.agent_positions[i * 2 + 1]), 0, MAP_SIZE - 1);
            const int ny = std::clamp(static_cast<int>(std::round(static_cast<float>(cy) + dy)), 0, MAP_SIZE - 1);
            const int nx = std::clamp(static_cast<int>(std::round(static_cast<float>(cx) + dx)), 0, MAP_SIZE - 1);

            const MoveEval move_eval = evaluate_move(s, env_idx, i, ny - cy, nx - cx);
            if (move_eval == MoveEval::BLOCKED)
                continue;

            const int tile_id = terrain_id_at(env_idx, ny, nx);
            const float speed_multiplier = std::max(0.0f, agent_spec(i, 10 + tile_id));
            const float speed = std::max(0.0f, agent_spec(i, 5) * speed_multiplier);
            s.agent_positions[i * 2] = std::clamp(s.agent_positions[i * 2] + dy * speed, 0.0f, MAP_MAX);
            s.agent_positions[i * 2 + 1] = std::clamp(s.agent_positions[i * 2 + 1] + dx * speed, 0.0f, MAP_MAX);

            if (move_eval == MoveEval::ALLOW_AND_STUCK)
                s.agent_stuck[i] = 1.0f;

            s.agent_battery[i] = std::max(0.0f, s.agent_battery[i] - 1.0f);
            if (s.agent_battery[i] <= 0.0f)
                s.agent_stuck[i] = 1.0f;

            rewards(env_idx, i) += 0.0f;
        }
    }

    void resolve_rescues(GameStateView &s)
    {
        for (int stuck_idx = 0; stuck_idx < N_AGENTS; ++stuck_idx)
        {
            if (s.agent_stuck[stuck_idx] < 0.5f)
                continue;
            const int sy = static_cast<int>(s.agent_positions[stuck_idx * 2]);
            const int sx = static_cast<int>(s.agent_positions[stuck_idx * 2 + 1]);

            for (int rescuer = 0; rescuer < N_AGENTS; ++rescuer)
            {
                if (rescuer == stuck_idx)
                    continue;
                if (s.agent_stuck[rescuer] > 0.5f || s.agent_battery[rescuer] <= 0.0f)
                    continue;

                const int ry = static_cast<int>(s.agent_positions[rescuer * 2]);
                const int rx = static_cast<int>(s.agent_positions[rescuer * 2 + 1]);
                if (ry == sy && rx == sx)
                {
                    s.agent_stuck[stuck_idx] = 0.0f;
                    s.agent_battery[stuck_idx] = std::max(1.0f, agent_spec(stuck_idx, 6) * 0.25f);
                    break;
                }
            }
        }
    }

    std::pair<int, int> update_pois_and_interactions(GameStateView &s, int env_idx)
    {
        std::uniform_int_distribution<int> dir_dist(0, 4);
        int newly_found = 0;
        int newly_saved = 0;

        for (int p = 0; p < n_pois; ++p)
        {
            if (s.poi_saved[p] > 0.5f)
                continue;

            if (poi_spec(p, 0) > 0.5f)
            {
                const int dir = dir_dist(rngs[env_idx]);
                int dy = 0;
                int dx = 0;
                if (dir == 1)
                    dy = -1;
                else if (dir == 2)
                    dy = 1;
                else if (dir == 3)
                    dx = -1;
                else if (dir == 4)
                    dx = 1;

                const float py = s.poi_positions[p * 2];
                const float px = s.poi_positions[p * 2 + 1];
                s.poi_positions[p * 2] = std::clamp(py + static_cast<float>(dy), 0.0f, MAP_MAX);
                s.poi_positions[p * 2 + 1] = std::clamp(px + static_cast<float>(dx), 0.0f, MAP_MAX);
            }

            const int py = static_cast<int>(s.poi_positions[p * 2]);
            const int px = static_cast<int>(s.poi_positions[p * 2 + 1]);
            const int allowed_mask = static_cast<int>(poi_spec(p, 1));

            for (int i = 0; i < N_AGENTS; ++i)
            {
                const int ay = static_cast<int>(s.agent_positions[i * 2]);
                const int ax = static_cast<int>(s.agent_positions[i * 2 + 1]);
                if (ay != py || ax != px)
                    continue;

                const int cls = static_cast<int>(agent_spec(i, 9));
                const bool can_save = (allowed_mask & (1 << cls)) != 0;

                if (can_save)
                {
                    if (s.poi_saved[p] < 0.5f)
                    {
                        s.poi_saved[p] = 1.0f;
                        newly_saved++;
                    }
                    s.poi_found[p] = 1.0f;
                }
                else if (s.poi_found[p] < 0.5f)
                {
                    s.poi_found[p] = 1.0f;
                    newly_found++;
                }
            }
        }

        return {newly_found, newly_saved};
    }

    int update_local_observations(GameStateView &s, int env_idx, std::array<float, N_AGENTS> *new_tile_credit)
    {
        int new_tiles = 0;

        for (int i = 0; i < N_AGENTS; ++i)
        {
            const int ay = std::clamp(static_cast<int>(s.agent_positions[i * 2]), 0, MAP_SIZE - 1);
            const int ax = std::clamp(static_cast<int>(s.agent_positions[i * 2 + 1]), 0, MAP_SIZE - 1);
            const float alt = altitude_at(s, ay, ax);
            const int vr = std::max(1, static_cast<int>(std::round(agent_spec(i, 6) * std::max(0.1f, alt))));
            s.agent_view_range[i] = static_cast<float>(vr);

            const int ys = std::max(0, ay - vr);
            const int ye = std::min(MAP_SIZE, ay + vr + 1);
            const int xs = std::max(0, ax - vr);
            const int xe = std::min(MAP_SIZE, ax + vr + 1);

            float *local_mask = s.local_obs_mask + i * FLAT_MAP_SIZE;
            for (int y = ys; y < ye; ++y)
            {
                const int row = y * MAP_SIZE;
                for (int x = xs; x < xe; ++x)
                {
                    const int idx = row + x;
                    local_mask[idx] = 1.0f;
                    if (s.global_obs_mask[idx] < 0.5f)
                    {
                        s.global_obs_mask[idx] = 1.0f;
                        undiscovered_remaining[env_idx]--;
                        new_tiles++;
                        if (new_tile_credit != nullptr)
                            (*new_tile_credit)[i] += 1.0f;
                    }
                }
            }

            float *my_local_agents = s.local_agent_layers + i * (N_AGENTS * FLAT_MAP_SIZE);
            for (int j = 0; j < N_AGENTS; ++j)
            {
                float *channel = my_local_agents + j * FLAT_MAP_SIZE;
                const int jy = std::clamp(static_cast<int>(s.agent_positions[j * 2]), 0, MAP_SIZE - 1);
                const int jx = std::clamp(static_cast<int>(s.agent_positions[j * 2 + 1]), 0, MAP_SIZE - 1);
                if (std::abs(jy - ay) <= vr && std::abs(jx - ax) <= vr)
                    channel[jy * MAP_SIZE + jx] = 1.0f;
            }

            float *my_poi = s.local_poi_layers + i * FLAT_MAP_SIZE;
            for (int p = 0; p < n_pois; ++p)
            {
                if (s.poi_saved[p] > 0.5f)
                    continue;
                const int py = std::clamp(static_cast<int>(s.poi_positions[p * 2]), 0, MAP_SIZE - 1);
                const int px = std::clamp(static_cast<int>(s.poi_positions[p * 2 + 1]), 0, MAP_SIZE - 1);
                if (std::abs(py - ay) <= vr && std::abs(px - ax) <= vr)
                    my_poi[py * MAP_SIZE + px] = 1.0f;
            }
        }

        return new_tiles;
    }

    std::string execute_radio(GameStateView &s, const std::array<int, N_AGENTS> &radio_actions)
    {
        std::ostringstream oss;
        for (int sender = 0; sender < N_AGENTS; ++sender)
        {
            if (radio_actions[sender] == 0)
                continue;

            oss << "agent_" << sender << " radio(" << radio_actions[sender] << "): POI/ally update\n";

            const float *sender_poi = s.local_poi_layers + sender * FLAT_MAP_SIZE;
            const float *sender_agents = s.local_agent_layers + sender * (N_AGENTS * FLAT_MAP_SIZE);

            const int sy = std::clamp(static_cast<int>(s.agent_positions[sender * 2]), 0, MAP_SIZE - 1);
            const int sx = std::clamp(static_cast<int>(s.agent_positions[sender * 2 + 1]), 0, MAP_SIZE - 1);
            const int sidx = sy * MAP_SIZE + sx;

            for (int recv = 0; recv < N_AGENTS; ++recv)
            {
                if (recv == sender)
                    continue;

                float *recv_poi = s.local_poi_layers + recv * FLAT_MAP_SIZE;
                for (int idx = 0; idx < FLAT_MAP_SIZE; ++idx)
                    if (sender_poi[idx] > 0.5f)
                        recv_poi[idx] = 1.0f;

                float *recv_sender_channel = s.local_agent_layers + recv * (N_AGENTS * FLAT_MAP_SIZE) + sender * FLAT_MAP_SIZE;
                std::memset(recv_sender_channel, 0, FLAT_MAP_SIZE * sizeof(float));
                recv_sender_channel[sidx] = 1.0f;

                for (int j = 0; j < N_AGENTS; ++j)
                {
                    const float *src = sender_agents + j * FLAT_MAP_SIZE;
                    float *dst = s.local_agent_layers + recv * (N_AGENTS * FLAT_MAP_SIZE) + j * FLAT_MAP_SIZE;
                    for (int idx = 0; idx < FLAT_MAP_SIZE; ++idx)
                        if (src[idx] > 0.5f)
                            dst[idx] = 1.0f;
                }
            }
        }
        return oss.str();
    }

    void rebuild_global_layers(GameStateView &s)
    {
        std::memset(s.global_unsaved_pois, 0, FLAT_MAP_SIZE * sizeof(float));
        for (int p = 0; p < n_pois; ++p)
        {
            if (s.poi_saved[p] > 0.5f)
                continue;
            const int py = std::clamp(static_cast<int>(s.poi_positions[p * 2]), 0, MAP_SIZE - 1);
            const int px = std::clamp(static_cast<int>(s.poi_positions[p * 2 + 1]), 0, MAP_SIZE - 1);
            s.global_unsaved_pois[py * MAP_SIZE + px] = 1.0f;
        }

        std::memset(s.global_agent_layers, 0, static_cast<size_t>(N_AGENTS) * FLAT_MAP_SIZE * sizeof(float));
        for (int i = 0; i < N_AGENTS; ++i)
        {
            const int ay = std::clamp(static_cast<int>(s.agent_positions[i * 2]), 0, MAP_SIZE - 1);
            const int ax = std::clamp(static_cast<int>(s.agent_positions[i * 2 + 1]), 0, MAP_SIZE - 1);
            s.global_agent_layers[i * FLAT_MAP_SIZE + ay * MAP_SIZE + ax] = 1.0f;
        }
    }

    bool all_pois_saved(const GameStateView &s) const
    {
        for (int p = 0; p < n_pois; ++p)
            if (s.poi_saved[p] < 0.5f)
                return false;
        return true;
    }
};

PYBIND11_MODULE(cpp_engine, m)
{
    REGISTER_FEATURE_TYPE_ENUM(m);

    py::class_<BatchedEnvironment>(m, "BatchedEnvironment")
        .def(py::init<
                 int,
                 int,
                 py::array_t<float, py::array::c_style | py::array::forcecast>,
                 py::array_t<float, py::array::c_style | py::array::forcecast>,
                 py::array_t<float, py::array::c_style | py::array::forcecast>,
                 py::array_t<float, py::array::c_style | py::array::forcecast>,
                 py::array_t<float, py::array::c_style | py::array::forcecast>,
                 py::array_t<int, py::array::c_style | py::array::forcecast>,
                 py::array_t<int, py::array::c_style | py::array::forcecast>,
                 py::array_t<int, py::array::c_style | py::array::forcecast>,
                 py::array_t<int, py::array::c_style | py::array::forcecast>,
                 bool,
                 float,
                 float,
                 float,
                 int>(),
             py::arg("n_envs"),
             py::arg("seed"),
             py::arg("terrain_tensor"),
             py::arg("agent_params"),
             py::arg("poi_params"),
             py::arg("initial_agent_pos"),
             py::arg("initial_poi_pos"),
             py::arg("tile_sup_walking"),
             py::arg("tile_sup_aquatic"),
             py::arg("tile_sup_flying"),
             py::arg("tile_is_blocking"),
             py::arg("cooperative_rewards") = true,
             py::arg("reward_new_tile") = 0.05f,
             py::arg("reward_found") = 2.0f,
             py::arg("reward_saved") = 20.0f,
             py::arg("max_frames") = 250)
        .def("reset", &BatchedEnvironment::reset)
        .def("reset_single", &BatchedEnvironment::reset_single, py::arg("env_idx"))
        .def("step", &BatchedEnvironment::step, py::arg("actions"))
        .def("get_action_mask", &BatchedEnvironment::get_action_mask)
        .def("get_memory_view", &BatchedEnvironment::get_memory_view)
        .def("get_state", &BatchedEnvironment::get_state)
        .def("get_stride", &BatchedEnvironment::get_stride)
        .def("get_flat_map_size", &BatchedEnvironment::get_flat_map_size)
        .def("get_terrain_channels", &BatchedEnvironment::get_terrain_channels)
        .def("get_num_pois", &BatchedEnvironment::get_num_pois)
        .def("radio_render", &BatchedEnvironment::radio_render)
        .def_readonly("num_envs", &BatchedEnvironment::num_envs);
}
