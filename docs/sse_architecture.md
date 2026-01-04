# SSE Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser Client                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  EventSource('/api/benchmark-sse?interval=5')          │   │
│  │                                                          │   │
│  │  addEventListener('status', updateVMStatus)             │   │
│  │  addEventListener('progress', updateProgress)           │   │
│  │  addEventListener('task_complete', addResult)           │   │
│  │  addEventListener('error', handleError)                 │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ▲                                     │
│                           │ SSE Stream                          │
│                           │ (text/event-stream)                 │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                           │  Local Server (port 8765)           │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │        GET /api/benchmark-sse?interval=5               │   │
│  │                                                          │   │
│  │  _stream_benchmark_updates(interval=5)                  │   │
│  │                                                          │   │
│  │  Loop every 5 seconds:                                  │   │
│  │    1. _fetch_background_tasks()                         │   │
│  │    2. send_event('status', vm_data)                     │   │
│  │    3. if vm_ready:                                      │   │
│  │         _detect_running_benchmark(vm_ip)                │   │
│  │         send_event('progress', progress_data)           │   │
│  │    4. Check benchmark_progress.json                     │   │
│  │    5. sleep(interval)                                   │   │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │    _detect_running_benchmark(vm_ip, container)          │   │
│  │                                                          │   │
│  │    1. SSH to vm_ip                                      │   │
│  │    2. docker exec {container} pgrep -f 'python.*run.py' │   │
│  │    3. tail -100 /tmp/waa_benchmark.log                  │   │
│  │    4. Parse log with regex                              │   │
│  │    5. Return {running, progress, current_task}          │   │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            │ SSH (port 22)
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      Azure VM (waa-eval-vm)                     │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │         Docker Container (winarena)                     │   │
│  │                                                          │   │
│  │  ┌────────────────────────────────────────────────┐    │   │
│  │  │  Windows 11 VM (QEMU)                          │    │   │
│  │  │                                                 │    │   │
│  │  │  ┌────────────────────────────────────────┐   │    │   │
│  │  │  │  WAA Flask Server (port 5000)          │   │    │   │
│  │  │  │  - /probe endpoint                     │   │    │   │
│  │  │  │  - Task execution                      │   │    │   │
│  │  │  └────────────────────────────────────────┘   │    │   │
│  │  │                                                 │    │   │
│  │  │  ┌────────────────────────────────────────┐   │    │   │
│  │  │  │  Benchmark Process                     │   │    │   │
│  │  │  │  python run.py                         │   │    │   │
│  │  │  │  └─> /tmp/waa_benchmark.log            │   │    │   │
│  │  │  └────────────────────────────────────────┘   │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                                                          │   │
│  │  VNC Server (port 8006)                                 │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Initial Connection

```
Browser                    Server
   │                         │
   ├──GET /api/benchmark-sse?interval=5
   │                         │
   │◄────200 OK─────────────┤
   │  Content-Type: text/event-stream
   │  Cache-Control: no-cache
   │                         │
```

### 2. Status Event

```
Server                     VM
   │                         │
   ├──_fetch_background_tasks()
   │                         │
   ├──Found docker_container task
   │                         │
   ├──event: status─────────►Browser
   data: {
     "type": "vm_status",
     "connected": true,
     "phase": "ready",
     "waa_ready": true,
     ...
   }
```

### 3. Progress Detection

```
Server                     VM
   │                         │
   ├──SSH azureuser@vm_ip───►│
   │                         │
   ├──docker exec winarena pgrep -f 'python.*run.py'
   │◄──PID: 1234─────────────┤
   │                         │
   ├──tail -100 /tmp/waa_benchmark.log
   │◄──Task 5/30, Step 12────┤
   │                         │
   ├──Parse log with regex──┤
   │                         │
   ├──event: progress───────►Browser
   data: {
     "tasks_completed": 5,
     "total_tasks": 30,
     "current_task": "task_001",
     "current_step": 12
   }
```

### 4. Task Completion Detection

```
Server
   │
   ├──Detect current_task changed
   │  from "task_001" to "task_002"
   │
   ├──event: task_complete──►Browser
   data: {
     "task_id": "task_001",
     "success": true,
     "score": 1.0
   }
   │
   ├──Update last_task = "task_002"
```

## Component Interaction

```
┌─────────────────┐
│  EventSource    │  Browser API
│  (Browser)      │  - Auto-reconnect
└────────┬────────┘  - Event parsing
         │           - Connection management
         │
         │ HTTP GET (Keep-Alive)
         │
┌────────▼────────┐
│ SSE Endpoint    │  /api/benchmark-sse
│ (local.py)      │  - Stream events
└────────┬────────┘  - Handle disconnects
         │           - Validate params
         │
         ├──────────────┬──────────────┐
         │              │              │
┌────────▼────────┐ ┌──▼──────────┐ ┌─▼──────────────┐
│ _fetch_        │ │_detect_      │ │benchmark_      │
│ background_    │ │running_      │ │progress.json   │
│ tasks()        │ │benchmark()   │ │(local file)    │
└────────┬────────┘ └──┬──────────┘ └────────────────┘
         │              │
         │              │ SSH
         │              │
    ┌────▼──────────────▼────┐
    │  Azure VM + Docker     │
    │  - VM status           │
    │  - Container status    │
    │  - Benchmark process   │
    │  - Log files           │
    └────────────────────────┘
```

## Event Timeline

```
Time    Event Type       Trigger                    Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0s      status          Initial poll                VM offline
5s      status          Interval poll               VM starting
10s     status          Interval poll               VM ready, WAA not ready
15s     status          Interval poll               VM ready, WAA ready
20s     status          Interval poll               VM ready, WAA ready
        progress        Benchmark detected          Task 1/30, Step 0
25s     status          Interval poll               VM ready, WAA ready
        progress        Benchmark running           Task 1/30, Step 5
30s     status          Interval poll               VM ready, WAA ready
        progress        Benchmark running           Task 1/30, Step 10
35s     status          Interval poll               VM ready, WAA ready
        progress        Benchmark running           Task 2/30, Step 0
        task_complete   Task changed (1→2)          task_001, success: true
40s     status          Interval poll               VM ready, WAA ready
        progress        Benchmark running           Task 2/30, Step 5
```

## Error Handling Flow

```
┌──────────────────┐
│  Client Request  │
└────────┬─────────┘
         │
         ▼
    ┌────────────────┐
    │ Validate       │◄──Invalid interval?──► Clamp to 1-60
    │ Parameters     │
    └────────┬───────┘
         │
         ▼
    ┌────────────────┐
    │ Set SSE Headers│
    │ Start Stream   │
    └────────┬───────┘
         │
         ▼
    ┌────────────────┐
    │ Fetch Tasks    │◄──Exception?──► Continue (empty data)
    └────────┬───────┘
         │
         ▼
    ┌────────────────┐
    │ SSH to VM      │◄──Timeout?──► Continue (running: false)
    └────────┬───────┘
         │
         ▼
    ┌────────────────┐
    │ Parse Logs     │◄──Parse error?──► Use defaults
    └────────┬───────┘
         │
         ▼
    ┌────────────────┐
    │ Send Event     │◄──Write error?──► Break (client gone)
    └────────┬───────┘
         │
         ▼
    ┌────────────────┐
    │ Sleep(interval)│
    └────────┬───────┘
         │
         └──────► Loop
```

## State Machine

```
┌─────────────┐
│   VM Off    │
└──────┬──────┘
       │ az vm start
       ▼
┌─────────────┐
│  Starting   │ phase: "booting"
└──────┬──────┘
       │ Boot complete
       ▼
┌─────────────┐
│   Docker    │ phase: "downloading"
│  Starting   │ phase: "building"
└──────┬──────┘
       │ Container up
       ▼
┌─────────────┐
│  Windows    │ phase: "oobe"
│  Installing │
└──────┬──────┘
       │ Setup scripts run
       ▼
┌─────────────┐
│   WAA Ready │ phase: "ready"
│             │ waa_ready: true
└──────┬──────┘
       │ Benchmark started
       ▼
┌─────────────┐
│  Benchmark  │ progress events
│  Running    │ task_complete events
└──────┬──────┘
       │ All tasks done
       ▼
┌─────────────┐
│  Complete   │ status: "idle"
└─────────────┘
```

## Performance Characteristics

### Polling (Old)

```
Time: 0s    5s    10s   15s   20s   25s   30s
      │     │     │     │     │     │     │
      ├─req─┤     │     │     │     │     │
      └─res─┤     │     │     │     │     │
            ├─req─┤     │     │     │     │
            └─res─┤     │     │     │     │
                  ├─req─┤     │     │     │
                  └─res─┤     │     │     │
                        ├─req─┤     │     │
                        └─res─┤     │     │

Requests: 6 in 30s
Overhead: 6 × (headers + empty response)
```

### SSE (New)

```
Time: 0s    5s    10s   15s   20s   25s   30s
      │     │     │     │     │     │     │
      ├─────────connection open──────────┤
      │     │     │     │     │     │     │
      └evt1─┴evt2─┴evt3─┴evt4─┴evt5─┴evt6┘

Requests: 1 in 30s
Overhead: 1 × headers + 6 × event
Events only when data available
```

## Scalability

```
┌────────────────────────────────────────┐
│           Load Balancer                │
│         (if needed)                    │
└──────┬─────────────────────┬───────────┘
       │                     │
       │                     │
┌──────▼──────┐       ┌──────▼──────┐
│  Server 1   │       │  Server 2   │
│  (8765)     │       │  (8765)     │
└──────┬──────┘       └──────┬──────┘
       │                     │
       └──────────┬──────────┘
                  │
           ┌──────▼──────┐
           │   Azure VM  │
           │  (shared)   │
           └─────────────┘

Note: Each server maintains independent SSE
connections. VM state is shared (SSH-based).
```

## Connection Lifecycle

```
1. Client Opens Connection
   ┌──────────────────────────┐
   │ new EventSource(url)     │
   └────────────┬─────────────┘
                │
                ▼
2. Server Accepts
   ┌──────────────────────────┐
   │ do_GET() matches path    │
   │ Send SSE headers         │
   │ Start streaming loop     │
   └────────────┬─────────────┘
                │
                ▼
3. Stream Events (Loop)
   ┌──────────────────────────┐
   │ while True:              │
   │   fetch data             │
   │   send events            │
   │   sleep(interval)        │
   └────────────┬─────────────┘
                │
                ▼
4. Disconnect
   ┌──────────────────────────┐
   │ Client: close()          │
   │ Server: write error      │
   │ Loop breaks              │
   └──────────────────────────┘
```

## Security Layers

```
┌──────────────────────────┐
│  Browser Same-Origin     │  CORS headers
│  Policy (CORS)           │  Access-Control-Allow-Origin
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  HTTPS (Production)      │  TLS encryption
│  Certificate validation  │  (not implemented yet)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Rate Limiting           │  Max connections per IP
│  (Future)                │  (not implemented yet)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Authentication          │  API key validation
│  (Future)                │  (not implemented yet)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  SSH Key Auth            │  ~/.ssh/id_rsa
│  to Azure VM             │  (existing)
└──────────────────────────┘
```

This architecture provides a scalable, efficient foundation for real-time benchmark monitoring while maintaining compatibility with existing systems.
