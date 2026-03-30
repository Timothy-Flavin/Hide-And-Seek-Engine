#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <cstring>
#include <algorithm>

// Define ssize_t for cross-platform compatibility (if not already handled)
#ifndef ssize_t
#ifdef _WIN32
typedef ptrdiff_t ssize_t;
#endif
#endif

// ---------------CONVENIENCE TENSOR STRIDE OBJECTS-------------------
// What individual agents with limited radio see
// There would be n_agent of these objects per environment
// or these strides will be added to env_stride to switch between
// environments
struct DecentralizedPartialObsStrides
{
    const ssize_t map_area;
    // Stride pointer  location for tile type layers
    const ssize_t TYLE_TYPE_START = 0; // n_tile_types * map_stride
    // Stride pointer  location for altitude layer
    const ssize_t ALTITUDE_TYPE_START; // w*h
    // Stride pointer location for person of interests who have been seen but not saved
    const ssize_t PIO_START;             // w*h
    const ssize_t OBSERVED_START;        // w*h
    const ssize_t MY_LOCATION_START;     // w*h
    const ssize_t OTHER_LOCATIONS_START; // (n_agents-1)*w*h
    DecentralizedPartialObsStrides(ssize_t w, ssize_t h, ssize_t n_tile_types, ssize_t n_agents)
        : map_area(w * h),
          TYLE_TYPE_START(0),
          ALTITUDE_TYPE_START(n_tile_types * map_area),
          PIO_START(ALTITUDE_TYPE_START + map_area),
          OBSERVED_START(PIO_START + map_area),
          MY_LOCATION_START(OBSERVED_START + map_area),
          OTHER_LOCATIONS_START(MY_LOCATION_START + map_area) {}
};
// What individual agents with centralized control / perfect radio see
struct CentralizedPartialObsStrides
{
    const ssize_t map_area;
    // Stride pointer  location for tile type layers
    const ssize_t TYLE_TYPE_START = 0; // n_tile_types * w*h
    // Stride pointer  location for altitude layer
    const ssize_t ALTITUDE_TYPE_START; // w*h
    // Stride pointer  location for person of interests who have been seen but not saved
    const ssize_t PIO_START;      // w*h
    const ssize_t OBSERVED_START; // w*h
    // All agents because this would be a centralized actor taking all agent actions
    const ssize_t AGENT_LOCATIONS_START; // n_agent_layers*w*h
    CentralizedPartialObsStrides(ssize_t w, ssize_t h, ssize_t n_tile_types, ssize_t n_agent_layers)
        : map_area(w * h),
          TYLE_TYPE_START(0),
          ALTITUDE_TYPE_START(n_tile_types * map_area),
          PIO_START(ALTITUDE_TYPE_START + map_area),
          OBSERVED_START(PIO_START + map_area),
          AGENT_LOCATIONS_START(OBSERVED_START + map_area) {}
};
// The true game state including hidden variables for centralized training
struct CentralizedStateStrides
{
    const ssize_t map_area;
    // Stride pointer location for tile type layers
    const ssize_t TYLE_TYPE_START = 0; // n_tile_types * w*h
    // Stride pointer  location for altitude layer
    const ssize_t ALTITUDE_TYPE_START; // w*h
    // Stride pointer true location for all person of interests who
    // are not saved. They need not be spotted yet
    const ssize_t PIO_START; // w*h
    // This will be one layer because only the true discovered map matters
    const ssize_t OBSERVED_START;        // w*h
    const ssize_t AGENT_LOCATIONS_START; // n_agent_layers*w*h
    CentralizedStateStrides(ssize_t w, ssize_t h, ssize_t n_tile_types, ssize_t n_agent_layers)
        : map_area(w * h),
          TYLE_TYPE_START(0),
          ALTITUDE_TYPE_START(n_tile_types * map_area),
          PIO_START(ALTITUDE_TYPE_START + map_area),
          OBSERVED_START(PIO_START + map_area),
          AGENT_LOCATIONS_START(OBSERVED_START + map_area) {}
};

//-------------------INTERNAL ENVIRONMENT DATA STRUCTURES------------------------

// Tile responsibilities like type altitude and observed
struct alignas(8) Tile
{
    uint32_t flags;
    float altitude;

    // --- Internal Constants ---
    static constexpr int TILE_TYPE_SHIFT = 5;
    static constexpr uint32_t TILE_TYPE_MASK = 0x7F;
    static constexpr int AGENT_MASK_SHIFT = 12;

    // ==========================================
    // TILE TYPE (Bits 5-11)
    // ==========================================
    inline uint32_t get_type() const
    {
        return (flags >> TILE_TYPE_SHIFT) & TILE_TYPE_MASK;
    }

    inline void set_type(uint32_t type_id)
    {
        flags = (flags & ~(TILE_TYPE_MASK << TILE_TYPE_SHIFT)) |
                ((type_id & TILE_TYPE_MASK) << TILE_TYPE_SHIFT);
    }

    // ==========================================
    // AGENT OBSERVED MASKS (Bits 12-31)
    // ==========================================
    inline bool has_agent_seen(int agent_id) const
    {
        return (flags >> (AGENT_MASK_SHIFT + agent_id)) & 1U;
    }

    inline void set_agent_seen(int agent_id)
    {
        flags |= (1U << (AGENT_MASK_SHIFT + agent_id));
    }

    inline void clear_agent_seen(int agent_id)
    {
        flags &= ~(1U << (AGENT_MASK_SHIFT + agent_id));
    }

    // ==========================================
    // PHYSICAL & GLOBAL PROPERTIES (Bits 0-4)
    // ==========================================
    inline bool is_walkable() const { return flags & 1U; }
    inline bool is_flyable() const { return (flags >> 1) & 1U; }
    inline bool is_aquatic() const { return (flags >> 2) & 1U; }
    inline bool is_blocking() const { return (flags >> 3) & 1U; }

    inline void set_walkable(bool v) { v ? flags |= 1U : flags &= ~1U; }
    inline void set_flyable(bool v) { v ? flags |= (1U << 1) : flags &= ~(1U << 1); }
    inline void set_aquatic(bool v) { v ? flags |= (1U << 2) : flags &= ~(1U << 2); }
    inline void set_blocking(bool v) { v ? flags |= (1U << 3) : flags &= ~(1U << 3); }

    // Clearly denotes this is the property of the *tile* being globally observed
    inline bool is_global_observed() const { return (flags >> 4) & 1U; }
    inline void set_global_observed() { flags |= (1U << 4); }
};

// 1. Logically grouped AoS structures (Alignments ensure clean cache lines)
struct alignas(16) AgentState
{
    float x;
    float y;
    float battery;
    float view_range;
    float deployment_remaining;
    int type;
    uint8_t stuck;
    // 3 bytes implicit padding
};

struct alignas(16) POIState
{
    float x;
    float y;
    uint32_t savable_by_mask; // Bitmask: (1 << agent_id) replaces bool array
    uint8_t found;
    uint8_t saved;
    uint8_t moves;
    // 1 byte implicit padding
};

// 2. The View Struct: Maps to the contiguous flat memory
struct EnvStateView
{
    Tile *grid;
    AgentState *agents;
    POIState *pois;
    float *agent_speeds; // Size: n_agents * n_tile_types
    int *current_frame;
    int *undiscovered_remaining;
};

class EnvironmentArena
{
public:
    std::vector<uint8_t> memory;
    size_t env_stride;

    int n_agents;
    int n_pois;
    int n_tile_types;
    int map_area;

    EnvironmentArena(int num_envs, int w, int h, int num_agents, int num_pois, int num_tile_types)
        : n_agents(num_agents),
          n_pois(num_pois),
          n_tile_types(num_tile_types),
          map_area(w * h)
    {
        // Calculate the exact memory footprint of each component for a single environment
        size_t grid_bytes = map_area * sizeof(Tile);
        size_t agent_bytes = n_agents * sizeof(AgentState);
        size_t poi_bytes = n_pois * sizeof(POIState);
        size_t speed_bytes = (n_agents * n_tile_types) * sizeof(float);
        size_t counter_bytes = 2 * sizeof(int);

        // Sum the exact required bytes with zero arbitrary internal padding.
        // Because Tile (8), AgentState (28), POIState (16), float (4), and int (4)
        // are all naturally aligned to at least 4 bytes, they will pack perfectly safely.
        size_t raw_stride = grid_bytes + agent_bytes + poi_bytes + speed_bytes + counter_bytes;

        // Align ONLY the total environment boundary to a 64-byte Cache Line.
        // This prevents Thread A (Environment 0) and Thread B (Environment 1)
        // from writing to the same physical L1/L2 cache line (False Sharing).
        env_stride = (raw_stride + 63) & ~63;

        // Allocate the contiguous memory block for all environments, initialized to 0.
        memory.resize(num_envs * env_stride, 0);
    }

    // Explicitly inline for hot-loop performance
    inline EnvStateView get_env_view(int env_idx)
    {
        EnvStateView view;

        // Calculate the starting memory address for this specific environment
        uint8_t *base = memory.data() + (env_idx * env_stride);
        size_t offset = 0;

        view.grid = reinterpret_cast<Tile *>(base + offset);
        offset += map_area * sizeof(Tile);

        view.agents = reinterpret_cast<AgentState *>(base + offset);
        offset += n_agents * sizeof(AgentState);

        view.pois = reinterpret_cast<POIState *>(base + offset);
        offset += n_pois * sizeof(POIState);

        view.agent_speeds = reinterpret_cast<float *>(base + offset);
        offset += (n_agents * n_tile_types) * sizeof(float);

        view.current_frame = reinterpret_cast<int *>(base + offset);
        offset += sizeof(int);

        view.undiscovered_remaining = reinterpret_cast<int *>(base + offset);

        return view;
    }
};