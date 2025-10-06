#!/bin/bash

# Specify the log file path
LOG_FILE="ram_usage.log"

# Infinite loop to monitor RAM usage every second
while true; do
    # Get RAM usage details from free command (in MB)
    RAM_USED=$(free -m | awk 'NR==2{print $3}')
    RAM_TOTAL=$(free -m | awk 'NR==2{print $2}')
    RAM_FREE=$(free -m | awk 'NR==2{print $4}')

    # Log the RAM usage with a timestamp to the file
    echo "$(date '+%Y-%m-%d %H:%M:%S') - RAM Used: ${RAM_USED}MB, Total: ${RAM_TOTAL}MB, Free: ${RAM_FREE}MB" >> "$LOG_FILE"

    # Wait for a second before the next iteration
    sleep 1
done
