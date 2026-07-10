#!/usr/bin/env python3
import time
import math
import random
import matplotlib.pyplot as plt
from dynamixel_sdk import *
# TODO: make random time intervals, slightly random movements 

DEVICE_NAME = "/dev/ttyUSB0"
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000

DXL_IDS = [0, 1, 2, 3]

TORQUE_ENABLE_ADDRS = [512, 512, 562, 562]
POS_ADDRESSES       = [564, 564, 596, 596]
PRESENT_POS_ADDRS   = [580, 580, 611, 611]

LEN_TORQUE_ENABLE  = 1
LEN_GOAL_POSITION  = 4
LEN_PRESENT_POS    = 4

MAX_DEG_PER_MOTOR = [179.82,179.82,179.91,180]
MAX_POS_PER_MOTOR = [303454,303454,2047,131593]
COUNTS_PER_DEGREE = [MAX_POS_PER_MOTOR[i] / MAX_DEG_PER_MOTOR[i] for i in range(4)]

# (Amplitude, Frequency, Phase)
COMBINATION_POOL = [
    (45.0, 0.50, 0.0),
    (60.0, 0.25, math.pi/2),
    (30.0, 0.80, math.pi),
    (90.0, 0.10, 0.0),
    (50.0, 0.40, math.pi/4),
    (75.0, 0.20, math.pi/3),
    (20.0, 1.00, math.pi/6),
    (40.0, 0.60, math.pi/8),
    (80.0, 0.15, math.pi),
    (25.0, 0.90, math.pi/2),
    (65.0, 0.30, 0.0),
    (35.0, 0.70, math.pi/4)
]

CENTERS = [0.0, 0.0, 10.0, -10.0]

CHANGE_INTERVAL = 10.0 
last_change_time = 0.0

current_assignments = [random.choice(COMBINATION_POOL) for _ in range(4)]

def degrees_to_position(degrees, idx):
    return int(round(degrees * COUNTS_PER_DEGREE[idx]))

def position_to_degrees(position, idx):
    if position & 0x80000000:
        position = position - 0x100000000
    return position / COUNTS_PER_DEGREE[idx]


portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

groupBulkWrite = GroupBulkWrite(portHandler, packetHandler)
groupBulkRead = GroupBulkRead(portHandler, packetHandler)

if portHandler.openPort() and portHandler.setBaudRate(BAUDRATE):
    print("Port opened and Baudrate configured successfully!")
else:
    print("Failed to initialize port/baudrate.")
    exit()

print("Configuring motors...")
for i in range(4):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_IDS[i], TORQUE_ENABLE_ADDRS[i], 1)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        print(f"Failed to enable torque for ID: {DXL_IDS[i]}")
        exit()

    dxl_addparam_result = groupBulkRead.addParam(DXL_IDS[i], PRESENT_POS_ADDRS[i], LEN_PRESENT_POS)
    if not dxl_addparam_result:
        print(f"[ID:{DXL_IDS[i]}] groupBulkRead addparam failed")
        exit()


time_history = []
pos_history = [[], [], [], []] 

print("\nStarting Random Sinusoidal Oscillations. Press Ctrl+C to stop, save data, and plot.")
print("Motors will randomly change combinations every 10 seconds.")
start_time = time.time()

try:
    while True:
        current_time = time.time() - start_time

        if current_time - last_change_time >= CHANGE_INTERVAL:
            current_assignments = [random.choice(COMBINATION_POOL) for _ in range(4)]
            last_change_time = current_time
            print(f"[{current_time:.1f}s] Randomly reassigned combinations!")

        for i in range(4):
            amp, freq, phase = current_assignments[i]
            target_deg = amp * math.sin(2 * math.pi * freq * current_time + phase) + CENTERS[i]
            target_position = degrees_to_position(target_deg, i)
            tx_position = target_position & 0xFFFFFFFF

            param_goal_position = [
                (tx_position >> 0) & 0xFF,
                (tx_position >> 8) & 0xFF,
                (tx_position >> 16) & 0xFF,
                (tx_position >> 24) & 0xFF
            ]
            groupBulkWrite.addParam(DXL_IDS[i], POS_ADDRESSES[i], LEN_GOAL_POSITION, param_goal_position)

        groupBulkWrite.txPacket()
        groupBulkWrite.clearParam()

        dxl_comm_result = groupBulkRead.txRxPacket()
        if dxl_comm_result == COMM_SUCCESS:
            time_history.append(current_time)

            for i in range(4):
                if groupBulkRead.isAvailable(DXL_IDS[i], PRESENT_POS_ADDRS[i], LEN_PRESENT_POS):
                    raw_pos = groupBulkRead.getData(DXL_IDS[i], PRESENT_POS_ADDRS[i], LEN_PRESENT_POS)
                    deg_pos = position_to_degrees(raw_pos, i)
                    pos_history[i].append(deg_pos)
                else:
                    pos_history[i].append(pos_history[i][-1] if pos_history[i] else CENTERS[i])

        time.sleep(0.01) # 100HZ not considering code exec

except KeyboardInterrupt:
    print("\nOscillation stopped. Processing data...")

finally:
    print("Disabling torque...")
    for i in range(4):
        packetHandler.write1ByteTxRx(portHandler, DXL_IDS[i], TORQUE_ENABLE_ADDRS[i], 0)
    portHandler.closePort()

    if time_history:
        log_filename = "motor_telemetry.txt"
        print(f"Saving data to '{log_filename}'...")
        try:
            with open(log_filename, "w") as f:
                # Header layout
                f.write("Time(s),Motor_0(deg),Motor_1(deg),Motor_2(deg),Motor_3(deg)\n")
                
                # Write row by row
                for idx in range(len(time_history)):
                    f.write(f"{time_history[idx]:.4f},"
                            f"{pos_history[0][idx]:.3f},"
                            f"{pos_history[1][idx]:.3f},"
                            f"{pos_history[2][idx]:.3f},"
                            f"{pos_history[3][idx]:.3f}\n")
            print("Data saved successfully!")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No data collected to save.")

    if time_history:
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b', 'm']
        for i in range(4):
            plt.plot(time_history, pos_history[i], color=colors[i], label=f'Motor ID {DXL_IDS[i]}')

        plt.title('Dynamixel PRO 4-Motor Randomized Sine Wave Tracking')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Present Position (Degrees)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.xlim(5.0, 15.0)
        print("Displaying graph. Close the window to terminate program entirely.")
        plt.show()
