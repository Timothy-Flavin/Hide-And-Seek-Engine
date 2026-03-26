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

#include <omp.h>

#include "gravity.h"

#ifndef ssize_t
#ifdef _WIN32
typedef ptrdiff_t ssize_t;
#endif
#endif

namespace py = pybind11;

namespace
{
constexpr int MOVE_ACTION_COUNT = 9;
constexpr std::array<int, MOVE_ACTION_COUNT> MOVE_DY = {0, -1, 1, 0, 0, -1, -1, 1, 1};
constexpr std::array<int, MOVE_ACTION_COUNT> MOVE_DX = {0, 0, 0, -1, 1, -1, 1, -1, 1};
}

struct GameStateView
{
    float *terrain_altitude;
    float *global_unsaved_pois;
    float *global_obs_mask;
    float *global_agent_layers;
    float *local_poi_layers;
    float *local_agent_layers;
    float *local_obs_mask;
    float *agent_positions;
    float *agent_deployment_remaining;
    float *agent_stuck;
    float *agent_view_range;
    float *agent_battery;
    float *poi_positions;
    float *poi_found;
    float *poi_saved;
};

class BatchedEnvironment
{
public:
    int num_envs;
    int seed;
    int n_tiles;
    int terrain_channels;
    int n_pois;
    int agent_spec_width;
    int poi_spec_width;
    int env_stride;
    bool cooperative_rewards;
    float reward_new_tile;
    float reward_found;
    float reward_saved;

    std::vector<std::mt19937> rngs;
    std::vector<float> data;
    std::vector<bool> env_terminated;
    std::vector<int> undiscovered_remaining;

    std::vector<float> terrain_templates;
    std::vector<int> terrain_ids;

    std::vector<float> agent_specs;
    std::vector<float> poi_specs;
    std::vector<float> init_agent_positions;
    std::vector<float> init_poi_positions;
    std::vector<int> tile_supports_walking;
    std::vector<int> tile_supports_aquatic;
    std::vector<int> tile_supports_flying;
    std::vector<int> tile_blocking;
    std::vector<std::string> radio_logs;

    BatchedEnvironment(
        int n_envs,
        int sim_seed,
        py::array_t<float, py::array::c_style | py::array::forcecast> terrain_tensor,
        py::array_t<float, py::array::c_style | py::array::forcecast> agent_params,
        py::array_t<float, py::array::c_style | py::array::forcecast> poi_params,
        py::array_t<float, py::array::c_style | py::array::forcecast> initial_agent_pos,
        py::array_t<float, py::array::c_style | py::array::forcecast> initial_poi_pos,
        py::array_t<int, py::array::c_style | py::array::forcecast> tile_sup_walking,
        py::array_t<int, py::array::c_style | py::array::forcecast> tile_sup_aquatic,
        py::array_t<int, py::array::c_style | py::array::forcecast> tile_sup_flying,
        py::array_t<int, py::array::c_style | py::array::forcecast> tile_is_blocking,
        bool coop_rewards = true,
        float reward_new_tile_val = 0.05f,
        float reward_found_val = 2.0f,
        float reward_saved_val = 20.0f)
        : num_envs(n_envs),
          seed(sim_seed),
          cooperative_rewards(coop_rewards),
          reward_new_tile(reward_new_tile_val),
          reward_found(reward_found_val),
          reward_saved(reward_saved_val)
    {
        if (num_envs <= 0)
            throw std::invalid_argument("n_envs must be > 0");

        auto tt = terrain_tensor.unchecked<4>();
        if (tt.shape(0) != num_envs || tt.shape(2) != MAP_SIZE || tt.shape(3) != MAP_SIZE)
            throw std::invalid_argument("terrain_tensor must have shape [num_envs, C, MAP_SIZE, MAP_SIZE]");
        terrain_channels = static_cast<int>(tt.shape(1));
        if (terrain_channels < 2)
            throw std::invalid_argument("terrain_tensor requires at least 2 channels (terrain one-hot + altitude)");
        n_tiles = terrain_channels - 1;

        auto ap = agent_params.unchecked<2>();
        if (ap.shape(0) != N_AGENTS)
            throw std::invalid_argument("agent_params must have shape [N_AGENTS, spec_width]");
        agent_spec_width = static_cast<int>(ap.shape(1));
        if (agent_spec_width < 10 + n_tiles)
            throw std::invalid_argument("agent spec width too small: expected >= 10 + n_tiles");

        auto pp = poi_params.unchecked<2>();
        n_pois = static_cast<int>(pp.shape(0));
        poi_spec_width = static_cast<int>(pp.shape(1));
        if (poi_spec_width < 2)
            throw std::invalid_argument("poi_params must have at least 2 columns [can_move, allowed_savers_mask]");

        auto iap = initial_agent_pos.unchecked<3>();
        if (iap.shape(0) != num_envs || iap.shape(1) != N_AGENTS || iap.shape(2) != 2)
            throw std::invalid_argument("initial_agent_pos must have shape [num_envs, N_AGENTS, 2]");

        auto ipp = initial_poi_pos.unchecked<3>();
        if (ipp.shape(0) != num_envs || ipp.shape(1) != n_pois || ipp.shape(2) != 2)
            throw std::invalid_argument("initial_poi_pos must have shape [num_envs, n_pois, 2]");

        auto sup_walk = tile_sup_walking.unchecked<1>();
        auto sup_aq = tile_sup_aquatic.unchecked<1>();
        auto sup_fl = tile_sup_flying.unchecked<1>();
        auto is_block = tile_is_blocking.unchecked<1>();
        if (sup_walk.shape(0) != n_tiles || sup_aq.shape(0) != n_tiles ||
            sup_fl.shape(0) != n_tiles || is_block.shape(0) != n_tiles)
            throw std::invalid_argument("tile support/blocking arrays must have shape [n_tiles]");

        const size_t terrain_count = static_cast<size_t>(num_envs) * terrain_channels * FLAT_MAP_SIZE;
        terrain_templates.resize(terrain_count);
        std::memcpy(terrain_templates.data(), terrain_tensor.data(), terrain_count * sizeof(float));

        terrain_ids.resize(static_cast<size_t>(num_envs) * FLAT_MAP_SIZE, 0);
        for (int e = 0; e < num_envs; ++e)
        {
            for (int idx = 0; idx < FLAT_MAP_SIZE; ++idx)
            {
                int best_tile = 0;
                float best_v = tt(e, 0, idx / MAP_SIZE, idx % MAP_SIZE);
                for (int t = 1; t < n_tiles; ++t)
                {
                    const float v = tt(e, t, idx / MAP_SIZE, idx % MAP_SIZE);
                    if (v > best_v)
                    {
                        best_v = v;
                        best_tile = t;
                    }
                }
                terrain_ids[static_cast<size_t>(e) * FLAT_MAP_SIZE + idx] = best_tile;
            }
        }

        agent_specs.resize(static_cast<size_t>(N_AGENTS) * agent_spec_width);
        std::memcpy(agent_specs.data(), agent_params.data(), agent_specs.size() * sizeof(float));

        poi_specs.resize(static_cast<size_t>(n_pois) * poi_spec_width);
        if (n_pois > 0)
            std::memcpy(poi_specs.data(), poi_params.data(), poi_specs.size() * sizeof(float));

        init_agent_positions.resize(static_cast<size_t>(num_envs) * N_AGENTS * 2);
        std::memcpy(init_agent_positions.data(), initial_agent_pos.data(), init_agent_positions.size() * sizeof(float));

        init_poi_positions.resize(static_cast<size_t>(num_envs) * n_pois * 2);
        if (n_pois > 0)
            std::memcpy(init_poi_positions.data(), initial_poi_pos.data(), init_poi_positions.size() * sizeof(float));

        tile_supports_walking.resize(n_tiles);
        tile_supports_aquatic.resize(n_tiles);
        tile_supports_flying.resize(n_tiles);
        tile_blocking.resize(n_tiles);
        for (int t = 0; t < n_tiles; ++t)
        {
            tile_supports_walking[t] = sup_walk(t);
            tile_supports_aquatic[t] = sup_aq(t);
            tile_supports_flying[t] = sup_fl(t);
            tile_blocking[t] = is_block(t);
        }

        env_stride =
            terrain_channels * FLAT_MAP_SIZE +
            FLAT_MAP_SIZE +
            FLAT_MAP_SIZE +
            (N_AGENTS * FLAT_MAP_SIZE) +
            (N_AGENTS * FLAT_MAP_SIZE) +
            (N_AGENTS * N_AGENTS * FLAT_MAP_SIZE) +
            (N_AGENTS * FLAT_MAP_SIZE) +
            (N_AGENTS * 2) +
            N_AGENTS +
            N_AGENTS +
            N_AGENTS +
            N_AGENTS +
            (n_pois * 2) +
            n_pois +
            n_pois;

        data.resize(static_cast<size_t>(num_envs) * env_stride, 0.0f);
        rngs.resize(num_envs);
        env_terminated.assign(num_envs, false);
        undiscovered_remaining.assign(num_envs, FLAT_MAP_SIZE);
        radio_logs.assign(num_envs, std::string());
        for (int i = 0; i < num_envs; ++i)
            rngs[i].seed(seed + i);

        reset();
    }

    void bind_state(GameStateView &s, int env_idx)
    {
        float *ptr = data.data() + static_cast<size_t>(env_idx) * env_stride;
        s.terrain_altitude = ptr;
        ptr += terrain_channels * FLAT_MAP_SIZE;
        s.global_unsaved_pois = ptr;
        ptr += FLAT_MAP_SIZE;
        s.global_obs_mask = ptr;
        ptr += FLAT_MAP_SIZE;
        s.global_agent_layers = ptr;
        ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.local_poi_layers = ptr;
        ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.local_agent_layers = ptr;
        ptr += N_AGENTS * N_AGENTS * FLAT_MAP_SIZE;
        s.local_obs_mask = ptr;
        ptr += N_AGENTS * FLAT_MAP_SIZE;
        s.agent_positions = ptr;
        ptr += N_AGENTS * 2;
        s.agent_deployment_remaining = ptr;
        ptr += N_AGENTS;
        s.agent_stuck = ptr;
        ptr += N_AGENTS;
        s.agent_view_range = ptr;
        ptr += N_AGENTS;
        s.agent_battery = ptr;
        ptr += N_AGENTS;
        s.poi_positions = ptr;
        ptr += n_pois * 2;
        s.poi_found = ptr;
        ptr += n_pois;
        s.poi_saved = ptr;
    }

    void reset_env(GameStateView &s, int e)
    {
        std::memset(data.data() + static_cast<size_t>(e) * env_stride, 0, env_stride * sizeof(float));

        const float *terrain_src = terrain_templates.data() + static_cast<size_t>(e) * terrain_channels * FLAT_MAP_SIZE;
        std::memcpy(s.terrain_altitude, terrain_src, static_cast<size_t>(terrain_channels) * FLAT_MAP_SIZE * sizeof(float));

        for (int i = 0; i < N_AGENTS; ++i)
        {
            const size_t base = static_cast<size_t>(e) * N_AGENTS * 2 + i * 2;
            s.agent_positions[i * 2] = std::clamp(init_agent_positions[base], 0.0f, MAP_MAX);
            s.agent_positions[i * 2 + 1] = std::clamp(init_agent_positions[base + 1], 0.0f, MAP_MAX);
            s.agent_deployment_remaining[i] = std::max(0.0f, agent_spec(i, 8));
            s.agent_stuck[i] = 0.0f;
            s.agent_view_range[i] = std::max(1.0f, agent_spec(i, 6));
            s.agent_battery[i] = std::max(1.0f, agent_spec(i, 7));
        }

        for (int p = 0; p < n_pois; ++p)
        {
            const size_t base = static_cast<size_t>(e) * n_pois * 2 + p * 2;
            s.poi_positions[p * 2] = std::clamp(init_poi_positions[base], 0.0f, MAP_MAX);
            s.poi_positions[p * 2 + 1] = std::clamp(init_poi_positions[base + 1], 0.0f, MAP_MAX);
            s.poi_found[p] = 0.0f;
            s.poi_saved[p] = 0.0f;
        }

        rebuild_global_layers(s);
        undiscovered_remaining[e] = FLAT_MAP_SIZE;
        update_local_observations(s, e, nullptr);
    }

    void reset()
    {
        env_terminated.assign(num_envs, false);
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            GameStateView s;
            bind_state(s, e);
            reset_env(s, e);
        }
    }

    void reset_single(int env_idx)
    {
        if (env_idx < 0 || env_idx >= num_envs)
            throw std::out_of_range("reset_single: env_idx out of range");
        GameStateView s;
        bind_state(s, env_idx);
        reset_env(s, env_idx);
        env_terminated[env_idx] = false;
    }

    std::pair<py::array_t<float>, py::array_t<bool>> step(py::array_t<float, py::array::c_style | py::array::forcecast> actions_array)
    {
        if (actions_array.ndim() != 2 && actions_array.ndim() != 3)
            throw std::invalid_argument("actions must have shape [E, A*3] or [E, A, 3]");
        if (actions_array.shape(0) != num_envs)
            throw std::invalid_argument("actions first dimension must match num_envs");

        py::array_t<float> rewards_array({num_envs, N_AGENTS});
        auto rewards = rewards_array.mutable_unchecked<2>();
        py::array_t<bool> terminated_array({num_envs});
        auto terminated = terminated_array.mutable_unchecked<1>();

        const float *act_data = actions_array.data();

#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            for (int i = 0; i < N_AGENTS; ++i)
                rewards(e, i) = 0.0f;

            if (env_terminated[e])
            {
                terminated(e) = true;
                continue;
            }

            GameStateView s;
            bind_state(s, e);

            std::array<int, N_AGENTS> radio_actions{};
            process_agent_movement(s, e, act_data, actions_array.ndim(), rewards, radio_actions);
            resolve_rescues(s);
            const auto poi_events = update_pois_and_interactions(s, e);

            std::array<float, N_AGENTS> new_tile_credit{};
            const int new_tiles = update_local_observations(s, e, &new_tile_credit);
            radio_logs[e] = execute_radio(s, radio_actions);
            rebuild_global_layers(s);

            if (cooperative_rewards)
            {
                const float total_reward =
                    static_cast<float>(new_tiles) * reward_new_tile +
                    static_cast<float>(poi_events.first) * reward_found +
                    static_cast<float>(poi_events.second) * reward_saved;
                for (int i = 0; i < N_AGENTS; ++i)
                    rewards(e, i) += total_reward;
            }
            else
            {
                for (int i = 0; i < N_AGENTS; ++i)
                    rewards(e, i) += new_tile_credit[i] * reward_new_tile;
            }

            const bool all_saved = all_pois_saved(s);
            terminated(e) = all_saved;
            env_terminated[e] = all_saved;
        }

        return {rewards_array, terminated_array};
    }

    py::array_t<bool> get_action_mask() const
    {
        py::array_t<bool> mask({num_envs, N_AGENTS, MOVE_ACTION_COUNT});
        auto m = mask.mutable_unchecked<3>();

#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            GameStateView s;
            const_cast<BatchedEnvironment *>(this)->bind_state(s, e);
            for (int i = 0; i < N_AGENTS; ++i)
            {
                for (int a = 0; a < MOVE_ACTION_COUNT; ++a)
                    m(e, i, a) = is_move_legal(s, e, i, MOVE_DY[a], MOVE_DX[a]);
            }
        }
        return mask;
    }

    std::pair<size_t, size_t> get_memory_view()
    {
        return {reinterpret_cast<size_t>(data.data()), data.size() * sizeof(float)};
    }

    py::array_t<float> get_state()
    {
        py::capsule base(data.data(), [](void *) {});
        return py::array_t<float>(
            {static_cast<ssize_t>(num_envs), static_cast<ssize_t>(env_stride)},
            {static_cast<ssize_t>(env_stride) * static_cast<ssize_t>(sizeof(float)),
             static_cast<ssize_t>(sizeof(float))},
            data.data(),
            base);
    }

    int get_stride() const { return env_stride; }
    int get_flat_map_size() const { return FLAT_MAP_SIZE; }
    int get_terrain_channels() const { return terrain_channels; }
    int get_num_pois() const { return n_pois; }

    void radio_render() const
    {
        for (int e = 0; e < num_envs; ++e)
        {
            if (!radio_logs[e].empty())
                std::cout << radio_logs[e];
        }
        std::cout.flush();
    }

private:
    float agent_spec(int agent_idx, int col) const
    {
        return agent_specs[static_cast<size_t>(agent_idx) * agent_spec_width + col];
    }

    float poi_spec(int poi_idx, int col) const
    {
        return poi_specs[static_cast<size_t>(poi_idx) * poi_spec_width + col];
    }

    float altitude_at(const GameStateView &s, int y, int x) const
    {
        const int idx = y * MAP_SIZE + x;
        const int altitude_channel = terrain_channels - 1;
        return s.terrain_altitude[altitude_channel * FLAT_MAP_SIZE + idx];
    }

    int terrain_id_at(int env_idx, int y, int x) const
    {
        const int idx = y * MAP_SIZE + x;
        return terrain_ids[static_cast<size_t>(env_idx) * FLAT_MAP_SIZE + idx];
    }

    enum class MoveEval
    {
        BLOCKED,
        ALLOW,
        ALLOW_AND_STUCK
    };

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

PYBIND11_MODULE(_core, m)
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
                 float>(),
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
             py::arg("reward_saved") = 20.0f)
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
