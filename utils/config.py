class Config:
    # Network topology parameters
    N_NODES = 50
    AVG_DEGREE = 4
    REWIRE_PROB = 0.1

    # DQN training parameters
    GAMMA = 0.99
    LR = 0.001
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 10
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64

    # Training parameters
    N_EPISODES = 1000
    PRINT_INTERVAL = 50
    N_TEST_SLICES = 10

    # Slice generation parameters
    SLICE_TYPES = ["URLLC", "eMBB", "mMTC"]
    SLICE_PROBABILITIES = [0.3, 0.5, 0.2]  # Higher probability for eMBB

    # QoS parameters
    URLLC_LATENCY = 1.0  # ms
    URLLC_EDGE_LATENCY = 0.5  # ms
    URLLC_BANDWIDTH = 100  # Mbps

    EMBB_LATENCY = 10.0  # ms
    EMBB_BANDWIDTH = 1000  # Mbps

    MMTC_LATENCY = 100.0  # ms
    MMTC_BANDWIDTH = 10  # Mbps

    # VNF parameters
    VNF_DELAY_RANGE = (0.1, 0.5)  # ms
    VNF_CPU_RANGE = (1, 4)  # vCPUs
    VNF_BW_RANGE = (10, 100)  # Mbps
