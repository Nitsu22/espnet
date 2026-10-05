# TSUBAME Resource Types and Point Costs

Verified on 2026-10-06 against the official [resource specifications](https://www.t4.cii.isct.ac.jp/docs/all/handbook.ja/jobs/#511), [resource coefficients](https://www.t4.cii.isct.ac.jp/fare_overview), and [charging rules, tables 2-4](https://www.somuka.titech.ac.jp/reiki_int/reiki_honbun/x385RG00001693.html). Recheck these sources when estimating a new run; update this reference if the published values change.

## Resource Types

Values are per resource unit. Memory is host RAM, not GPU memory. Specify units with `-l <type>=<count>`.

| Resource | CPU cores | Host RAM (GB) | GPUs | Point coefficient |
|---|---:|---:|---:|---:|
| `node_f` | 192 | 768 | 4 | 1.000 |
| `node_h` | 96 | 384 | 2 | 0.500 |
| `node_q` | 48 | 192 | 1 | 0.250 |
| `node_o` | 24 | 96 | 1/2 | 0.125 |
| `gpu_1` | 8 | 96 | 1 | 0.200 |
| `gpu_h` | 4 | 48 | 1/2 | 0.100 |
| `cpu_160` | 160 | 368 | 0 | 0.600 |
| `cpu_80` | 80 | 184 | 0 | 0.300 |
| `cpu_40` | 40 | 92 | 0 | 0.150 |
| `cpu_16` | 16 | 36.8 | 0 | 0.060 |
| `cpu_8` | 8 | 18.4 | 0 | 0.030 |
| `cpu_4` | 4 | 9.2 | 0 | 0.015 |

- CPU-only preparation or scoring: choose a `cpu_*` size that fits both memory and useful parallelism. For example, `cpu_8=1` fits up to 8 CPU cores and 18.4 GB host RAM; use a larger CPU allocation if either requirement exceeds that. If no CPU-only type meets a single-process memory requirement, explain that constraint when selecting a larger resource; memory across nodes cannot be pooled automatically for one process.
- One full GPU: compare `gpu_1=1` with `node_q=1`. Prefer `gpu_1` when 8 CPU cores and 96 GB host RAM suffice; `node_q` provides more CPU and memory if needed. Compare expected total points if the smaller CPU allocation would slow the GPU job.
- Two or four GPUs on one node: consider `node_h=1` or `node_f=1`, respectively, when the requested computation benefits from those GPU counts. Do not treat `node_q=4` as interchangeable with `node_f=1`; verify placement and multi-node support before requesting multiple units.
- Fractional GPU allocations (`node_o`, `gpu_h`): verify the actual GPU memory and application compatibility before choosing them.
- A job can request multiple units of one resource type; different resource types cannot be combined in one request. Split CPU and GPU phases into separate jobs when worthwhile, and verify prerequisite artifacts before starting the next phase.

## Estimate and Reduce Points

For an ordinary group-charged, usage-based job, estimate:

```text
points = units * resource_coefficient * priority_coefficient
         * (0.7 * max(actual_seconds, 300) + 0.1 * requested_h_rt_seconds)
         / 3600
```

The official rules round down to 0.0001 points. Priority `-5` has coefficient 1; `-4` and `-3` have coefficients 2 and 4, respectively. Use `-5` unless the user prioritizes faster scheduling over point minimization. This formula does not apply to reservations, subscription jobs, or the dedicated interactive queue.

- Compare the total across all required phases using credible runtime estimates. Increasing resource units or priority multiplies costs; extra parallelism helps only when its runtime reduction justifies the increase.
- Requested `h_rt` contributes to the final charge even when the job finishes early. Set a measured runtime estimate plus a modest margin, rather than a long default or a limit so short that it causes incomplete work and reruns.
- The actual-runtime term has a 300-second minimum for each charged job. Avoid unnecessary tiny jobs or repeated failing submissions; group compatible short operations when that reduces overhead and preserves the workflow.
- Inspect whether each stage actually uses CUDA before assigning it to a CPU job. Match stage boundaries, `--ngpu`, CPU workers, and threads to the allocation; changing the resource request alone does not make a GPU-dependent command CPU-compatible.
- Reuse verified completed data preparation or statistics when the requested experiment permits it, rather than repeating CPU work while reserving GPUs. The existing uncharged-trial eligibility rules still apply; point minimization does not authorize research or measurement through uncharged trials.
