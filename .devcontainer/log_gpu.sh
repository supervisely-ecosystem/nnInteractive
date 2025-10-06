# #!/bin/bash

# # Specify the log file path
# LOG_FILE="gpu_processes.log"

# # Infinite loop to monitor processes using GPU every second
# while true; do
#     # Get the count of processes using the GPU, excluding the header line
#     GPU_PROCESS_COUNT=$(nvidia-smi pmon -c 1 | tail -n +3 | wc -l)
    
#     # Log the count with a timestamp to the file
#     echo "$(date '+%Y-%m-%d %H:%M:%S') - GPU Processes, $GPU_PROCESS_COUNT" >> "$LOG_FILE"
    
#     # Wait for a second before the next iteration
#     sleep 1
# done
#!/bin/bash

# Файл лога
LOG_FILE="gpu_memory.log"

# Заголовок разъяснение полей
echo "# timestamp | gpu_index | gpu_mem_used_MiB | gpu_mem_total_MiB | per_process(pid:name:used_MiB;...)" >> "$LOG_FILE"

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"

  # По каждому GPU: используемая и общая память (MiB), без хедера и единиц
  # Пример строки: "0, 312, 24576"
  while IFS=, read -r GPU_IDX MEM_USED MEM_TOTAL; do
    GPU_IDX="$(echo "$GPU_IDX" | xargs)"
    MEM_USED="$(echo "$MEM_USED" | xargs)"
    MEM_TOTAL="$(echo "$MEM_TOTAL" | xargs)"

    # По‑процессам для всех GPU: pid, process_name, used_memory (MiB), без хедера и единиц
    # Пример строк: "1234, python, 2048"
    PROC_LINES="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"

    # Отфильтруем по конкретному GPU, если нужно MIG/мульти‑GPU: базовый nvidia-smi не всегда помечает GPU для каждой строки,
    # поэтому берём все процессы как сводный список и логируем целиком для простоты.
    # При необходимости, можно дополнительно парсить `nvidia-smi -i $GPU_IDX` в связке с pmon.
    PROC_AGG=""
    if [ -n "$PROC_LINES" ]; then
      # Сформируем компактный список "pid:name:MiB" через ';'
      PROC_AGG="$(echo "$PROC_LINES" | awk -F',' '{pid=$1; name=$2; mem=$3; gsub(/^ +| +$/,"",pid); gsub(/^ +| +$/,"",name); gsub(/^ +| +$/,"",mem); printf "%s%s:%s:%s", (NR==1?"":";"), pid, name, mem }')"
    fi

    echo "$TS | $GPU_IDX | $MEM_USED | $MEM_TOTAL | ${PROC_AGG:-none}" >> "$LOG_FILE"
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

  sleep 1
done
