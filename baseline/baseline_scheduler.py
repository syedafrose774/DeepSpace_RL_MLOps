class BaselineScheduler:
    def __init__(self):
        pass
        
    def select_action(self, state):
        # State:
        # [battery, signal, storage, high_prio, low_prio, comm_window, health]
        # Actions:
        # 0: Transmit high-priority data
        # 1: Transmit low-priority data
        # 2: Conserve power
        # 3: Enter low-power mode
        
        # unnormalize logic: actually the state passed here is normalized
        # but we can check if > 0
        high_prio = state[3]
        low_prio = state[4]
        comm_window = state[5]
        
        if comm_window == 1.0:
            if high_prio > 0:
                return 0
            elif low_prio > 0:
                return 1
            else:
                return 2
        else:
            return 2
