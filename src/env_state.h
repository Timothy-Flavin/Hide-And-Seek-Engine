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
struct Tile
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
struct AgentState
{
    float x;                    // 4 bytes
    float y;                    // 4 bytes
    float view_range;           // 4 bytes
    float deployment_remaining; // 4 bytes
    uint16_t last_x;            // 2 bytes
    uint16_t last_y;            // 2 bytes
    uint8_t stuck;              // 1 byte
    uint8_t _padding[3];        // 3 bytes explicitly added to reach 24
};

struct POIState
{
    float x;
    float y;
    // Bits 0-19: savable_by_mask
    // Bit 20:    found
    // Bit 21:    saved
    // Bit 22:    moves (bool: does this POI move?)
    uint32_t state;
    uint16_t last_x;
    uint16_t last_y;
    // --- Accessors ---
    // Agent Mask (0-19)
    void set_savable(int id, bool v) { v ? state |= (1u << id) : state &= ~(1u << id); }
    bool is_savable_by(int id) const { return (state >> id) & 1u; }

    // Logic Flags (20-22)
    void set_found(bool v) { v ? state |= (1u << 20) : state &= ~(1u << 20); }
    bool is_found() const { return (state >> 20) & 1u; }

    void set_saved(bool v) { v ? state |= (1u << 21) : state &= ~(1u << 21); }
    bool is_saved() const { return (state >> 21) & 1u; }

    void set_moves(bool v) { v ? state |= (1u << 22) : state &= ~(1u << 22); }
    bool does_move() const { return (state >> 22) & 1u; }
};

// What Agent A knows about Agent B
// AgentKnowledge: 14 bytes of data + 2 explicit padding bytes = 16 bytes total.
struct AgentKnowledge
{
    float x;                 // 4 bytes
    float y;                 // 4 bytes
    uint16_t last_x;         // 2 bytes
    uint16_t last_y;         // 2 bytes
    uint8_t has_contact;     // 1 byte: is this agent getting an update this frame
    uint8_t has_encountered; // 1 byte: has this agent ever gotten an update
    uint8_t is_stuck;        // 1 byte
    uint8_t _padding;        // 1 bytes explicitly added to reach 16
};

// What Agent A knows about POI (Survivor) K
struct POIKnowledge
{
    float x;             // 4 bytes
    float y;             // 4 bytes
    uint16_t last_x;     // 2 bytes
    uint16_t last_y;     // 2 bytes
    uint8_t knows_found; // 1 byte
    uint8_t knows_saved; // 1 byte
    uint8_t _padding[2]; // 2 bytes explicitly added to reach 16
};

// 2. The View Struct: Maps to the contiguous flat memory
struct EnvStateView
{
    Tile *grid; // spatial part of the observation
    AgentState *agents;
    POIState *pois;
    // --- NEW BELIEF MATRICES ---
    // Size: n_agents * n_agents. Index via [viewer_id * n_agents + target_id]
    AgentKnowledge *agent_knowledge; // used to fill agent layer(s) of tensor
    // Size: n_agents * n_pois. Index via [viewer_id * n_pois + poi_id]
    POIKnowledge *poi_knowledge; // used to fill poi layer of tensor
    float *agent_speeds;         // Size: n_agents * n_tile_types
    int *current_frame;
    int *undiscovered_remaining;
    int *poi_left;
    int *agents_left;
};

enum Mode
{
    DECENTRALIZED,
    CENTRALIZED,
    NO_OBS
};

class EnvironmentArena
{
private:
    // Helper to align memory offsets to specific boundaries (used only for env boundaries now)
    static constexpr size_t align_up(size_t val, size_t alignment)
    {
        return (val + alignment - 1) & ~(alignment - 1);
    }

public:
    std::vector<uint8_t> memory;
    uint8_t *aligned_base;
    size_t env_stride;
    Mode mode;

    int n_agents;
    int n_pois;
    int n_tile_types;
    int map_area;

    // Cache internal offset locations for blazing fast get_env_view calls
    size_t offset_agents, offset_pois, offset_speeds, offset_agent_know, offset_poi_know, offset_counters;

    EnvironmentArena(int num_envs, int w, int h, int num_agents, int num_pois, int num_tile_types, Mode mode_value, int init_mode)
        : n_agents(num_agents), n_pois(num_pois), n_tile_types(num_tile_types), map_area(w * h), mode(mode_value)
    {
        size_t current_offset = 0;

        // Grid (starts at 0)
        current_offset += map_area * sizeof(Tile);

        // Tightly pack the arrays back-to-back.
        // No internal padding needed because all sizes are multiples of 4.
        offset_agents = current_offset;
        current_offset += n_agents * sizeof(AgentState);

        offset_pois = current_offset;
        current_offset += n_pois * sizeof(POIState);

        offset_speeds = current_offset;
        current_offset += (n_agents * n_tile_types) * sizeof(float);

        offset_agent_know = current_offset;
        if (mode == Mode::DECENTRALIZED)
        {
            current_offset += (n_agents * n_agents) * sizeof(AgentKnowledge);
        }

        offset_poi_know = current_offset;
        if (mode == Mode::DECENTRALIZED)
        {
            current_offset += (n_agents * n_pois) * sizeof(POIKnowledge);
        }

        offset_counters = current_offset;
        current_offset += 4 * sizeof(int);

        // ONLY align the total environment footprint to a 256-byte boundary.
        // This isolates threads writing to different environments.
        env_stride = align_up(current_offset, 256);

        // Allocate block (+255 to allow manual 256-byte alignment of the base pointer)
        memory.resize(num_envs * env_stride + 255);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(memory.data());
        aligned_base = reinterpret_cast<uint8_t *>(align_up(raw_ptr, 256));

        if (init_mode == 1)
        {
            std::memset(aligned_base, 0, num_envs * env_stride);
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (int e = 0; e < num_envs; ++e)
            {
                std::memset(aligned_base + (e * env_stride), 0, env_stride);
            }
        }
    }

    // Fully inline, utilizing pre-calculated offsets for minimum overhead
    inline EnvStateView get_env_view(int env_idx)
    {
        EnvStateView view;
        uint8_t *base = aligned_base + (env_idx * env_stride);

        view.grid = reinterpret_cast<Tile *>(base);
        view.agents = reinterpret_cast<AgentState *>(base + offset_agents);
        view.pois = reinterpret_cast<POIState *>(base + offset_pois);
        view.agent_speeds = reinterpret_cast<float *>(base + offset_speeds);

        if (mode == Mode::DECENTRALIZED)
        {
            view.agent_knowledge = reinterpret_cast<AgentKnowledge *>(base + offset_agent_know);
            view.poi_knowledge = reinterpret_cast<POIKnowledge *>(base + offset_poi_know);
        }
        else
        {
            view.agent_knowledge = nullptr;
            view.poi_knowledge = nullptr;
        }

        view.current_frame = reinterpret_cast<int *>(base + offset_counters);
        view.undiscovered_remaining = reinterpret_cast<int *>(base + offset_counters + sizeof(int));
        view.poi_left = reinterpret_cast<int *>(base + offset_counters + 2 * sizeof(int));
        view.agents_left = reinterpret_cast<int *>(base + offset_counters + 3 * sizeof(int));

        return view;
    }
};