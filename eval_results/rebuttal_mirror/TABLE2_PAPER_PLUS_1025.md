**Pooled iCEM success rates (%) across tasks and compute tiers, three training runs.** Each entry reports mean ± SD over set-level measurements. LeWM, SCALE and Aux: 18 measurements (three independently trained runs — the two runs of the original submission and one new run — each evaluated on six held-out sets of 100 episodes). DINO-WM: 12 measurements (the two runs of the original submission). Overall averages T1–T5 within each measurement. Bold marks the best method per column within each task.

| Task | Method | T1 | T2 | T3 | T4 | T5 | Overall |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 85.8±2.4 | 80.4±2.7 | 73.7±5.8 | 55.2±4.7 | 25.6±4.8 | 64.1±3.0 |
| Push-T | SCALE | 89.3±2.6 | 86.0±2.9 | **79.7±4.1** | 60.2±6.0 | 25.7±5.7 | 68.2±2.9 |
| Push-T | Aux | **89.7±2.4** | **86.2±2.6** | 79.5±4.3 | **61.6±5.5** | **26.9±5.0** | **68.8±2.9** |
| Push-T | DINO-WM | 70.0±4.5 | 65.3±4.5 | 58.8±5.1 | 41.5±8.8 | 23.1±3.7 | 51.7±4.3 |
| Reacher | LeWM | 85.3±2.8 | 84.9±3.5 | 84.8±3.3 | 79.3±5.8 | 61.9±4.0 | 79.2±2.1 |
| Reacher | SCALE | **86.2±3.8** | 85.5±3.9 | **86.6±3.2** | **80.4±3.6** | **63.2±3.1** | **80.4±2.5** |
| Reacher | Aux | 84.9±3.7 | **86.4±2.6** | 85.7±3.4 | 80.2±3.7 | 63.0±4.5 | 80.0±1.9 |
| Reacher | DINO-WM | 78.7±4.9 | 78.2±4.8 | 80.8±3.5 | 76.0±4.0 | 53.0±5.5 | 73.3±3.1 |
| Cube | LeWM | 69.3±4.8 | 67.6±3.9 | 69.1±4.3 | 62.1±4.2 | 55.7±5.3 | 64.8±4.0 |
| Cube | SCALE | **76.8±3.7** | **76.6±3.9** | **75.6±2.8** | **65.9±3.0** | **59.5±4.4** | **70.9±3.0** |
| Cube | Aux | 74.1±3.5 | 72.4±4.7 | 72.7±3.5 | 64.9±3.6 | 57.6±3.7 | 68.3±3.3 |
| Cube | DINO-WM | 69.9±2.5 | 67.4±1.6 | 66.0±1.7 | 57.7±3.8 | 51.4±3.6 | 62.5±1.3 |
| Two-Room | LeWM | 90.9±1.5 | 93.0±1.6 | 88.7±1.9 | 77.8±2.5 | 62.0±2.7 | 82.5±1.0 |
| Two-Room | SCALE | 99.9±0.3 | 99.7±0.7 | 97.8±1.2 | 90.0±3.4 | 76.6±2.6 | 92.8±1.0 |
| Two-Room | Aux | 92.5±1.9 | 93.4±1.7 | 88.5±2.2 | 78.7±2.3 | 62.9±3.1 | 83.2±1.1 |
| Two-Room | DINO-WM | **100.0±0.0** | **100.0±0.0** | **99.8±0.5** | **98.6±0.8** | **92.5±2.1** | **98.2±0.5** |
| PointMaze | LeWM | 83.2±4.0 | 82.2±4.3 | 80.6±3.9 | 78.8±5.3 | 73.0±4.1 | 79.6±3.6 |
| PointMaze | SCALE | 87.4±3.3 | 85.8±4.4 | 83.9±3.0 | 80.7±4.3 | 76.1±3.9 | 82.8±2.5 |
| PointMaze | Aux | 83.5±5.9 | 82.6±6.3 | 82.9±7.5 | 82.0±5.0 | 77.3±6.4 | 81.6±5.6 |
| PointMaze | DINO-WM | **93.4±2.6** | **92.2±2.6** | **90.8±3.4** | **90.3±2.6** | **85.3±2.6** | **90.4±2.1** |

**Pooled CEM success rates (%) across tasks and compute tiers, three training runs.** Each entry reports mean ± SD over set-level measurements. LeWM, SCALE and Aux: 18 measurements (three independently trained runs — the two runs of the original submission and one new run — each evaluated on six held-out sets of 100 episodes). DINO-WM: 12 measurements (the two runs of the original submission). Overall averages T1–T5 within each measurement. Bold marks the best method per column within each task.

| Task | Method | T1 | T2 | T3 | T4 | T5 | Overall |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.6±2.4 | 89.8±2.9 | 80.3±5.2 | 58.0±5.1 | 34.4±4.5 | 70.8±2.8 |
| Push-T | SCALE | **95.0±2.9** | 92.2±2.9 | 83.9±3.2 | 62.9±6.8 | 33.0±5.6 | 73.4±2.8 |
| Push-T | Aux | 94.3±2.4 | **92.8±2.1** | **84.9±4.3** | **64.7±6.0** | **34.6±6.1** | **74.2±3.2** |
| Push-T | DINO-WM | 74.8±4.6 | 72.1±4.5 | 65.3±4.9 | 46.1±4.6 | 24.3±6.1 | 56.5±4.0 |
| Reacher | LeWM | 82.5±1.8 | 79.2±4.1 | 73.4±4.4 | 65.6±5.0 | 56.1±5.7 | 71.3±2.6 |
| Reacher | SCALE | 80.9±4.1 | **79.8±5.1** | 74.1±4.6 | 66.1±5.2 | **57.2±4.8** | 71.6±2.7 |
| Reacher | Aux | **82.7±4.4** | 79.7±4.3 | **74.8±3.0** | **66.3±4.9** | 56.6±6.7 | **72.0±3.2** |
| Reacher | DINO-WM | 77.2±2.8 | 74.8±5.4 | 70.8±5.9 | 61.8±3.4 | 50.3±7.3 | 67.0±3.0 |
| Cube | LeWM | 68.2±4.8 | 66.2±3.7 | 60.6±4.2 | 56.4±3.5 | 54.1±4.0 | 61.1±3.4 |
| Cube | SCALE | **74.7±4.1** | **71.3±4.0** | **64.4±3.5** | 55.6±4.0 | **55.0±4.1** | **64.2±3.2** |
| Cube | Aux | 71.7±4.5 | 69.6±4.2 | 63.1±5.0 | **56.7±3.9** | **55.0±4.3** | 63.2±3.5 |
| Cube | DINO-WM | 73.4±3.1 | 66.9±2.3 | 58.4±4.4 | 51.6±4.5 | 53.9±3.8 | 60.9±2.9 |
| Two-Room | LeWM | 87.7±2.7 | 87.2±2.7 | 82.6±3.6 | 71.9±2.7 | 58.0±3.3 | 77.5±1.5 |
| Two-Room | SCALE | 99.6±0.6 | 99.3±0.8 | 95.0±2.3 | 83.2±2.5 | 70.5±4.0 | 89.5±0.8 |
| Two-Room | Aux | 88.4±2.3 | 87.5±2.8 | 83.2±3.2 | 71.3±3.1 | 59.1±3.5 | 77.9±1.2 |
| Two-Room | DINO-WM | **100.0±0.0** | **99.8±0.4** | **98.3±1.2** | **94.1±2.2** | **85.8±3.7** | **95.6±1.0** |
| PointMaze | LeWM | 85.0±4.5 | 85.3±4.1 | 85.3±4.4 | 83.2±3.4 | 74.8±5.0 | 82.7±3.2 |
| PointMaze | SCALE | 86.9±3.3 | 87.4±3.8 | 85.9±4.1 | 83.9±4.1 | 78.1±3.8 | 84.4±2.7 |
| PointMaze | Aux | 87.3±5.2 | 86.7±5.6 | 85.7±6.0 | 82.2±5.2 | 78.8±5.0 | 84.1±4.6 |
| PointMaze | DINO-WM | **98.2±1.2** | **98.3±1.4** | **97.2±1.3** | **92.4±3.0** | **87.2±2.9** | **94.7±1.4** |

LaTeX (same numbers):

```latex
% iCEM
\begin{tabular}{llcccccc}
\toprule
Task & Method & T1 & T2 & T3 & T4 & T5 & Overall \\
\midrule
\multirow{4}{*}{Push-T} & LeWM & 85.8$\pm$2.4 & 80.4$\pm$2.7 & 73.7$\pm$5.8 & 55.2$\pm$4.7 & 25.6$\pm$4.8 & 64.1$\pm$3.0 \\
 & SCALE & 89.3$\pm$2.6 & 86.0$\pm$2.9 & \textbf{79.7$\pm$4.1} & 60.2$\pm$6.0 & 25.7$\pm$5.7 & 68.2$\pm$2.9 \\
 & Aux & \textbf{89.7$\pm$2.4} & \textbf{86.2$\pm$2.6} & 79.5$\pm$4.3 & \textbf{61.6$\pm$5.5} & \textbf{26.9$\pm$5.0} & \textbf{68.8$\pm$2.9} \\
 & DINO-WM & 70.0$\pm$4.5 & 65.3$\pm$4.5 & 58.8$\pm$5.1 & 41.5$\pm$8.8 & 23.1$\pm$3.7 & 51.7$\pm$4.3 \\
\midrule
\multirow{4}{*}{Reacher} & LeWM & 85.3$\pm$2.8 & 84.9$\pm$3.5 & 84.8$\pm$3.3 & 79.3$\pm$5.8 & 61.9$\pm$4.0 & 79.2$\pm$2.1 \\
 & SCALE & \textbf{86.2$\pm$3.8} & 85.5$\pm$3.9 & \textbf{86.6$\pm$3.2} & \textbf{80.4$\pm$3.6} & \textbf{63.2$\pm$3.1} & \textbf{80.4$\pm$2.5} \\
 & Aux & 84.9$\pm$3.7 & \textbf{86.4$\pm$2.6} & 85.7$\pm$3.4 & 80.2$\pm$3.7 & 63.0$\pm$4.5 & 80.0$\pm$1.9 \\
 & DINO-WM & 78.7$\pm$4.9 & 78.2$\pm$4.8 & 80.8$\pm$3.5 & 76.0$\pm$4.0 & 53.0$\pm$5.5 & 73.3$\pm$3.1 \\
\midrule
\multirow{4}{*}{Cube} & LeWM & 69.3$\pm$4.8 & 67.6$\pm$3.9 & 69.1$\pm$4.3 & 62.1$\pm$4.2 & 55.7$\pm$5.3 & 64.8$\pm$4.0 \\
 & SCALE & \textbf{76.8$\pm$3.7} & \textbf{76.6$\pm$3.9} & \textbf{75.6$\pm$2.8} & \textbf{65.9$\pm$3.0} & \textbf{59.5$\pm$4.4} & \textbf{70.9$\pm$3.0} \\
 & Aux & 74.1$\pm$3.5 & 72.4$\pm$4.7 & 72.7$\pm$3.5 & 64.9$\pm$3.6 & 57.6$\pm$3.7 & 68.3$\pm$3.3 \\
 & DINO-WM & 69.9$\pm$2.5 & 67.4$\pm$1.6 & 66.0$\pm$1.7 & 57.7$\pm$3.8 & 51.4$\pm$3.6 & 62.5$\pm$1.3 \\
\midrule
\multirow{4}{*}{Two-Room} & LeWM & 90.9$\pm$1.5 & 93.0$\pm$1.6 & 88.7$\pm$1.9 & 77.8$\pm$2.5 & 62.0$\pm$2.7 & 82.5$\pm$1.0 \\
 & SCALE & 99.9$\pm$0.3 & 99.7$\pm$0.7 & 97.8$\pm$1.2 & 90.0$\pm$3.4 & 76.6$\pm$2.6 & 92.8$\pm$1.0 \\
 & Aux & 92.5$\pm$1.9 & 93.4$\pm$1.7 & 88.5$\pm$2.2 & 78.7$\pm$2.3 & 62.9$\pm$3.1 & 83.2$\pm$1.1 \\
 & DINO-WM & \textbf{100.0$\pm$0.0} & \textbf{100.0$\pm$0.0} & \textbf{99.8$\pm$0.5} & \textbf{98.6$\pm$0.8} & \textbf{92.5$\pm$2.1} & \textbf{98.2$\pm$0.5} \\
\midrule
\multirow{4}{*}{PointMaze} & LeWM & 83.2$\pm$4.0 & 82.2$\pm$4.3 & 80.6$\pm$3.9 & 78.8$\pm$5.3 & 73.0$\pm$4.1 & 79.6$\pm$3.6 \\
 & SCALE & 87.4$\pm$3.3 & 85.8$\pm$4.4 & 83.9$\pm$3.0 & 80.7$\pm$4.3 & 76.1$\pm$3.9 & 82.8$\pm$2.5 \\
 & Aux & 83.5$\pm$5.9 & 82.6$\pm$6.3 & 82.9$\pm$7.5 & 82.0$\pm$5.0 & 77.3$\pm$6.4 & 81.6$\pm$5.6 \\
 & DINO-WM & \textbf{93.4$\pm$2.6} & \textbf{92.2$\pm$2.6} & \textbf{90.8$\pm$3.4} & \textbf{90.3$\pm$2.6} & \textbf{85.3$\pm$2.6} & \textbf{90.4$\pm$2.1} \\
\bottomrule
\end{tabular}

% CEM
\begin{tabular}{llcccccc}
\toprule
Task & Method & T1 & T2 & T3 & T4 & T5 & Overall \\
\midrule
\multirow{4}{*}{Push-T} & LeWM & 91.6$\pm$2.4 & 89.8$\pm$2.9 & 80.3$\pm$5.2 & 58.0$\pm$5.1 & 34.4$\pm$4.5 & 70.8$\pm$2.8 \\
 & SCALE & \textbf{95.0$\pm$2.9} & 92.2$\pm$2.9 & 83.9$\pm$3.2 & 62.9$\pm$6.8 & 33.0$\pm$5.6 & 73.4$\pm$2.8 \\
 & Aux & 94.3$\pm$2.4 & \textbf{92.8$\pm$2.1} & \textbf{84.9$\pm$4.3} & \textbf{64.7$\pm$6.0} & \textbf{34.6$\pm$6.1} & \textbf{74.2$\pm$3.2} \\
 & DINO-WM & 74.8$\pm$4.6 & 72.1$\pm$4.5 & 65.3$\pm$4.9 & 46.1$\pm$4.6 & 24.3$\pm$6.1 & 56.5$\pm$4.0 \\
\midrule
\multirow{4}{*}{Reacher} & LeWM & 82.5$\pm$1.8 & 79.2$\pm$4.1 & 73.4$\pm$4.4 & 65.6$\pm$5.0 & 56.1$\pm$5.7 & 71.3$\pm$2.6 \\
 & SCALE & 80.9$\pm$4.1 & \textbf{79.8$\pm$5.1} & 74.1$\pm$4.6 & 66.1$\pm$5.2 & \textbf{57.2$\pm$4.8} & 71.6$\pm$2.7 \\
 & Aux & \textbf{82.7$\pm$4.4} & 79.7$\pm$4.3 & \textbf{74.8$\pm$3.0} & \textbf{66.3$\pm$4.9} & 56.6$\pm$6.7 & \textbf{72.0$\pm$3.2} \\
 & DINO-WM & 77.2$\pm$2.8 & 74.8$\pm$5.4 & 70.8$\pm$5.9 & 61.8$\pm$3.4 & 50.3$\pm$7.3 & 67.0$\pm$3.0 \\
\midrule
\multirow{4}{*}{Cube} & LeWM & 68.2$\pm$4.8 & 66.2$\pm$3.7 & 60.6$\pm$4.2 & 56.4$\pm$3.5 & 54.1$\pm$4.0 & 61.1$\pm$3.4 \\
 & SCALE & \textbf{74.7$\pm$4.1} & \textbf{71.3$\pm$4.0} & \textbf{64.4$\pm$3.5} & 55.6$\pm$4.0 & \textbf{55.0$\pm$4.1} & \textbf{64.2$\pm$3.2} \\
 & Aux & 71.7$\pm$4.5 & 69.6$\pm$4.2 & 63.1$\pm$5.0 & \textbf{56.7$\pm$3.9} & \textbf{55.0$\pm$4.3} & 63.2$\pm$3.5 \\
 & DINO-WM & 73.4$\pm$3.1 & 66.9$\pm$2.3 & 58.4$\pm$4.4 & 51.6$\pm$4.5 & 53.9$\pm$3.8 & 60.9$\pm$2.9 \\
\midrule
\multirow{4}{*}{Two-Room} & LeWM & 87.7$\pm$2.7 & 87.2$\pm$2.7 & 82.6$\pm$3.6 & 71.9$\pm$2.7 & 58.0$\pm$3.3 & 77.5$\pm$1.5 \\
 & SCALE & 99.6$\pm$0.6 & 99.3$\pm$0.8 & 95.0$\pm$2.3 & 83.2$\pm$2.5 & 70.5$\pm$4.0 & 89.5$\pm$0.8 \\
 & Aux & 88.4$\pm$2.3 & 87.5$\pm$2.8 & 83.2$\pm$3.2 & 71.3$\pm$3.1 & 59.1$\pm$3.5 & 77.9$\pm$1.2 \\
 & DINO-WM & \textbf{100.0$\pm$0.0} & \textbf{99.8$\pm$0.4} & \textbf{98.3$\pm$1.2} & \textbf{94.1$\pm$2.2} & \textbf{85.8$\pm$3.7} & \textbf{95.6$\pm$1.0} \\
\midrule
\multirow{4}{*}{PointMaze} & LeWM & 85.0$\pm$4.5 & 85.3$\pm$4.1 & 85.3$\pm$4.4 & 83.2$\pm$3.4 & 74.8$\pm$5.0 & 82.7$\pm$3.2 \\
 & SCALE & 86.9$\pm$3.3 & 87.4$\pm$3.8 & 85.9$\pm$4.1 & 83.9$\pm$4.1 & 78.1$\pm$3.8 & 84.4$\pm$2.7 \\
 & Aux & 87.3$\pm$5.2 & 86.7$\pm$5.6 & 85.7$\pm$6.0 & 82.2$\pm$5.2 & 78.8$\pm$5.0 & 84.1$\pm$4.6 \\
 & DINO-WM & \textbf{98.2$\pm$1.2} & \textbf{98.3$\pm$1.4} & \textbf{97.2$\pm$1.3} & \textbf{92.4$\pm$3.0} & \textbf{87.2$\pm$2.9} & \textbf{94.7$\pm$1.4} \\
\bottomrule
\end{tabular}

```
