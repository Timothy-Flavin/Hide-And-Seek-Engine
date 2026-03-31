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
    float *torch_tensor_base = nullptr;
    float *torch_state_tensor_base = nullptr;
    Mode mode;

    // Stride helper objects (conditionally instantiated)
    std::unique_ptr<DecentralizedPartialObsStrides> decentral_obs_strides;
    std::unique_ptr<CentralizedPartialObsStrides> central_obs_strides;
    std::unique_ptr<CentralizedStateStrides> central_state_strides;

    int width;
    int height;
    std::vector<float> individual_rewards;
    std::vector<float> reward;
    std::vector<EnvStateView> env_views;

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
            agent.current_tile = static_cast<uint16_t>(final_y * width + final_x);
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
                for (int x = min_x; x <= max_x; ++x)
                {
                    float dist_sq = (agent.y - y) * (agent.y - y) + (agent.x - x) * (agent.x - x);
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

    std::string execute_radio(int e, const int *radio_act_data)
    {
        // for environment e:
        // If an agent chose radio action 0, it is messaging itself and
        // nothing happens. If it choses another action it is messaging
        // one of the other agents. It will share it's current information
        // updating it's x y in the other agent's tensor observation
        if (mode != Mode::DECENTRALIZED)
            return ""; // No-op for global and no_obs modes

        EnvStateView view = env_views[e];
        std::string log = "";

        for (int a = 0; a < n_agents; ++a)
        {
            int target_agent = radio_act_data[e * n_agents + a];

            // Assume action maps directly to agent ID (0 = agent 0, 1 = agent 1...)
            // If the environment defines a specific "0 = none, 1 = agent 0" shift, adjust target_agent accordingly.
            if (target_agent == a || target_agent < 0 || target_agent >= n_agents)
                continue;

            // Agent 'a' sends info to 'target_agent'

            // 1. Share 'a' location with 'target_agent'
            AgentKnowledge &target_ak = view.agent_knowledge[target_agent * n_agents + a];
            target_ak.x = view.agents[a].x;
            target_ak.y = view.agents[a].y;
            target_ak.has_contact = 1;

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
            // 3. Share Tile Knowledge (Merge maps)
            for (int i = 0; i < map_size; ++i)
            {
                Tile &t = view.grid[i];
                if (t.has_agent_seen(a))
                {
                    t.set_agent_seen(target_agent);
                }
            }
            log += "Agent " + std::to_string(a) + " -> " + std::to_string(target_agent) + "; ";
        }
        return log;
    }

    void update_battery_and_counters(e)
    {
        EnvStateView view = env_views[e];
        for (int a = 0; a < n_agents; ++a)
        {
            // If an agent is past it's battery capacity its' deployement ends
            if (view.agents[a].deployement > 0.0)
            {
                view.agents[a].deployement -= delta_time;
                if (view.agents[a].deployement < 0.0)
                {
                    --view.agents_left;
                }
            }
        }
        int poi_left = 0;
        for (int p = 0; p < n_pois; ++p)
        {
            if (!view.pois[p].saved)
                ++poi_left;
        }
        view.poi_left = poi_left;
    }

    void fill_torch_obs(e)
    {
    }

    void fill_torch_state(e)
    {
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
        std::vector<bool> saveable_rules,     // n_poi * n_agents can this poi be saved by this agent
        std::vector<float> initial_agent_pos, // n_agent * 2 x y start locs
        std::vector<float> initial_poi_pos,   // n_poi * 2 x y start locs
        uintptr_t tensor_ptr,                 // PyTorch pinned tensor pointer mapped via data_ptr() (Observations)
        bool requires_state = false,          // Should we allocate the true state tensor?
        uintptr_t state_tensor_ptr = 0,       // PyTorch pinned tensor pointer mapped via data_ptr() (Global State)
        bool coop_rewards = true,
        float reward_new_tile_val = 0.05f,
        float reward_found_val = 2.0f,
        float reward_saved_val = 20.0f,
        int max_frames_val = 250,
        int mode_value = 0)
        : num_envs(n_envs),
          seed(sim_seed),
          width(w),
          height(h),
          supports_walking(std::move(supports_walk)),
          supports_aquatic(std::move(supports_aqua)),
          supports_flying(std::move(supports_fly)),
          is_blocking(std::move(is_block)),
          type_map(std::move(t_map)),
          saveable_rules(std::move(save_map)),
          init_agent_positions(std::move(initial_agent_pos)),
          init_poi_positions(std::move(initial_poi_pos)),
          cooperative_rewards(coop_rewards),
          reward_new_tile(reward_new_tile_val),
          reward_found(reward_found_val),
          reward_saved(reward_saved_val),
          max_frames(max_frames_val),
          map_size(w * h),
          mode(static_cast<Mode>(mode_value)
    {
        if (mode_value < 0 || mode_value > 2)
        {
            throw(std::invalid_argument("mode value < 0 or > 2 cannot be bound into Mode enum"));
        }
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
        arena = std::make_unique<EnvironmentArena>(num_envs, width, height, n_agents, n_pois, n_tiles, mode);

        // 4. Bind the required tensor stride objects based on configuration
        bind_state(mode, requires_state, state_tensor_ptr, tensor_ptr);

        // 2. Pre-allocate the vector to prevent reallocations
        env_views.reserve(num_envs);
        // 3. Pre-cache all the views
        for (int e = 0; e < num_envs; ++e)
        {
            // get_env_view returns by value, push_back copies that 48-byte struct into the vector safely
            env_views.push_back(arena->get_env_view(e));
        }

        // 5. Fill the initial game state structure
        for (int env_idx = 0; env_idx < num_envs; ++env_idx)
        {
            reset_env(env_idx);
        }
    }

    void bind_state(const std::string &mode, bool requires_state, uintptr_t state_tensor_ptr, uintptr_t tensor_ptr)
    {
        if (mode == Mode::CENTRALIZED)
        {
            central_obs_strides = std::make_unique<CentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
            torch_spatial_tensor_base = reinterpret_cast<float *>(tensor_ptr)
        }
        else if (mode == Mode::DECENTRALIZED) // default to decentralized
        {
            decentral_obs_strides = std::make_unique<DecentralizedPartialObsStrides>(width, height, n_tiles, n_agents);
            torch_spatial_tensor_base = reinterpret_cast<float *>(tensor_ptr)
        }

        if (requires_state)
        {
            central_state_strides = std::make_unique<CentralizedStateStrides>(width, height, n_tiles, n_agents);
            torch_state_tensor_base = reinterpret_cast<float *>(state_tensor_ptr);
        }
    }

    void reset_env(int env_idx)
    {
        EnvStateView view = env_views[env_idx];

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

        view.poi_left = n_pois;
        view.agents_left = n_agents;
    }

    void reset()
    {
#pragma omp parallel for schedule(static)
        for (int e = 0; e < num_envs; ++e)
        {
            reset_env(e);
        }
    }

    void step(py::array_t<float, py::array::c_style | py::array::forcecast> move_actions_array, py::array_t<int, py::array::c_style> radio_actions_array)
    {
        reward = 0;
        rewards.fill(0);
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
            radio_logs[e] = execute_radio(e, radio_act_data);

            update_battery_and_counters(e);

            const bool all_saved = *(env_views[e].poi_left);
            const bool all_out_of_battery = *(env_views[e].agents_left);
            const bool timeout = current_frames[e] >= max_frames;
            env_terminated[e] = all_saved || all_out_of_battery;
            env_truncated[e] = timeout;

            // If we are using this environment for machine learning
            // then fill the torch buffers accordingly
            if (return_obs)
                fill_torch_obs(e);
            if (requires_state)
                fill_torch_state(e);
        }

        return {reward, individual_rewards, terminated, truncated};
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
