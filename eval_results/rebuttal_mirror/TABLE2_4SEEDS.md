**Pooled iCEM success rates (%) across tasks and compute tiers, four training runs.** Each entry reports mean ± SD over set-level measurements. LeWM, SCALE and Aux: 24 measurements (four independently trained runs — the two runs of the original submission and two new runs — each evaluated on six held-out sets of 100 episodes). DINO-WM: 12 measurements (the two runs of the original submission). Overall averages T1–T5 within each measurement. Bold marks the best method per column within each task.

| Task | Method | T1 | T2 | T3 | T4 | T5 | Overall |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.1±2.4 | 81.2±3.0 | 73.9±5.2 | 56.0±4.5 | 25.8±4.5 | 64.6±2.8 |
| Push-T | SCALE | 89.4±2.6 | 85.5±2.9 | 79.4±4.5 | 60.0±5.4 | 26.3±6.2 | 68.1±3.1 |
| Push-T | Aux | **89.7±2.4** | **86.2±2.5** | **80.0±4.4** | **61.8±5.1** | **27.0±5.0** | **68.9±2.8** |
| Push-T | DINO-WM | 70.0±4.5 | 65.3±4.5 | 58.8±5.1 | 41.5±8.8 | 23.1±3.7 | 51.7±4.3 |
| Reacher | LeWM | 85.2±2.5 | 84.3±3.7 | 85.0±3.2 | 79.4±5.2 | 61.7±4.0 | 79.1±1.9 |
| Reacher | SCALE | **85.8±3.9** | 84.8±3.9 | **85.8±3.7** | **80.2±3.1** | **63.0±3.2** | **79.9±2.5** |
| Reacher | Aux | 84.9±3.5 | **85.9±2.9** | **85.8±3.4** | 80.1±3.5 | 62.5±4.5 | 79.8±1.8 |
| Reacher | DINO-WM | 78.7±4.9 | 78.2±4.8 | 80.8±3.5 | 76.0±4.0 | 53.0±5.5 | 73.3±3.1 |
| Cube | LeWM | 68.9±4.4 | 67.4±3.7 | 69.0±3.9 | 61.8±4.3 | 55.8±5.1 | 64.6±3.8 |
| Cube | SCALE | **76.8±3.5** | **76.6±3.8** | **75.1±2.8** | **65.7±2.9** | **59.3±4.2** | **70.7±2.9** |
| Cube | Aux | 74.0±3.7 | 73.0±4.7 | 73.0±3.5 | 64.7±3.4 | 57.9±3.8 | 68.5±3.3 |
| Cube | DINO-WM | 69.9±2.5 | 67.4±1.6 | 66.0±1.7 | 57.7±3.8 | 51.4±3.6 | 62.5±1.3 |
| Two-Room | LeWM | 90.7±1.6 | 92.8±1.6 | 88.3±2.1 | 77.5±2.5 | 61.6±3.0 | 82.2±1.2 |
| Two-Room | SCALE | 99.9±0.3 | 99.7±0.6 | 97.9±1.2 | 90.3±3.2 | 76.7±2.5 | 92.9±1.0 |
| Two-Room | Aux | 92.4±2.0 | 93.4±1.6 | 88.4±2.0 | 78.6±2.4 | 62.5±2.8 | 83.1±1.1 |
| Two-Room | DINO-WM | **100.0±0.0** | **100.0±0.0** | **99.8±0.5** | **98.6±0.8** | **92.5±2.1** | **98.2±0.5** |
| PointMaze | LeWM | 79.2±8.3 | 78.2±8.2 | 77.8±6.4 | 77.3±5.9 | 73.2±4.0 | 77.1±5.6 |
| PointMaze | SCALE | 84.4±6.3 | 82.8±6.7 | 80.0±8.0 | 77.7±7.3 | 73.5±6.5 | 79.7±6.3 |
| PointMaze | Aux | 85.0±5.8 | 83.4±5.9 | 83.5±6.9 | 81.5±4.7 | 76.4±6.5 | 82.0±5.1 |
| PointMaze | DINO-WM | **93.4±2.6** | **92.2±2.6** | **90.8±3.4** | **90.3±2.6** | **85.3±2.6** | **90.4±2.1** |

**Pooled CEM success rates (%) across tasks and compute tiers, four training runs.** Each entry reports mean ± SD over set-level measurements. LeWM, SCALE and Aux: 24 measurements (four independently trained runs — the two runs of the original submission and two new runs — each evaluated on six held-out sets of 100 episodes). DINO-WM: 12 measurements (the two runs of the original submission). Overall averages T1–T5 within each measurement. Bold marks the best method per column within each task.

| Task | Method | T1 | T2 | T3 | T4 | T5 | Overall |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.8±2.6 | 90.0±2.8 | 80.7±4.7 | 58.2±4.9 | 33.5±5.3 | 70.9±2.9 |
| Push-T | SCALE | **95.1±2.7** | 92.2±2.9 | 84.4±3.3 | 62.7±6.2 | 33.7±5.6 | 73.6±2.8 |
| Push-T | Aux | 94.5±2.3 | **92.6±1.9** | **84.9±4.2** | **64.5±5.4** | **34.4±5.5** | **74.2±2.9** |
| Push-T | DINO-WM | 74.8±4.6 | 72.1±4.5 | 65.3±4.9 | 46.1±4.6 | 24.3±6.1 | 56.5±4.0 |
| Reacher | LeWM | 81.8±3.1 | 79.1±3.9 | 73.4±4.1 | 65.5±4.8 | 56.4±6.0 | 71.2±2.5 |
| Reacher | SCALE | 81.1±4.6 | **79.8±4.7** | 73.9±4.2 | 65.8±5.1 | **56.8±4.9** | 71.5±2.5 |
| Reacher | Aux | **82.4±4.1** | **79.8±3.8** | **74.1±3.9** | **67.3±4.9** | 56.5±6.4 | **72.0±3.0** |
| Reacher | DINO-WM | 77.2±2.8 | 74.8±5.4 | 70.8±5.9 | 61.8±3.4 | 50.3±7.3 | 67.0±3.0 |
| Cube | LeWM | 67.8±4.6 | 66.4±3.9 | 60.4±4.2 | 56.5±3.6 | 54.5±3.9 | 61.1±3.4 |
| Cube | SCALE | **74.7±3.8** | **71.6±3.8** | **64.7±3.7** | 55.4±4.0 | **55.2±4.0** | **64.3±3.1** |
| Cube | Aux | 71.8±4.4 | 69.4±3.8 | 63.9±4.8 | **56.6±3.7** | 54.9±3.9 | 63.3±3.2 |
| Cube | DINO-WM | 73.4±3.1 | 66.9±2.3 | 58.4±4.4 | 51.6±4.5 | 53.9±3.8 | 60.9±2.9 |
| Two-Room | LeWM | 87.3±2.7 | 86.3±3.0 | 82.3±3.7 | 71.8±2.5 | 58.0±3.0 | 77.1±1.5 |
| Two-Room | SCALE | 99.5±0.6 | 99.1±1.0 | 94.9±2.2 | 83.2±2.3 | 70.8±3.6 | 89.5±0.8 |
| Two-Room | Aux | 88.2±2.3 | 87.2±2.8 | 82.8±3.1 | 71.3±3.2 | 59.5±3.4 | 77.8±1.2 |
| Two-Room | DINO-WM | **100.0±0.0** | **99.8±0.4** | **98.3±1.2** | **94.1±2.2** | **85.8±3.7** | **95.6±1.0** |
| PointMaze | LeWM | 84.5±4.1 | 84.0±4.6 | 83.8±5.0 | 81.4±4.7 | 75.8±5.4 | 81.9±3.3 |
| PointMaze | SCALE | 83.6±7.0 | 83.8±7.5 | 83.0±6.7 | 80.9±7.0 | 75.7±6.1 | 81.4±6.3 |
| PointMaze | Aux | 87.8±4.7 | 87.4±5.2 | 86.3±5.4 | 83.1±5.5 | 78.0±4.9 | 84.5±4.2 |
| PointMaze | DINO-WM | **98.2±1.2** | **98.3±1.4** | **97.2±1.3** | **92.4±3.0** | **87.2±2.9** | **94.7±1.4** |

LaTeX (same numbers):

```latex
% iCEM
\begin{tabular}{llcccccc}
\toprule
Task & Method & T1 & T2 & T3 & T4 & T5 & Overall \\
\midrule
\multirow{4}{*}{Push-T} & LeWM & 86.1$\pm$2.4 & 81.2$\pm$3.0 & 73.9$\pm$5.2 & 56.0$\pm$4.5 & 25.8$\pm$4.5 & 64.6$\pm$2.8 \\
 & SCALE & 89.4$\pm$2.6 & 85.5$\pm$2.9 & 79.4$\pm$4.5 & 60.0$\pm$5.4 & 26.3$\pm$6.2 & 68.1$\pm$3.1 \\
 & Aux & \textbf{89.7$\pm$2.4} & \textbf{86.2$\pm$2.5} & \textbf{80.0$\pm$4.4} & \textbf{61.8$\pm$5.1} & \textbf{27.0$\pm$5.0} & \textbf{68.9$\pm$2.8} \\
 & DINO-WM & 70.0$\pm$4.5 & 65.3$\pm$4.5 & 58.8$\pm$5.1 & 41.5$\pm$8.8 & 23.1$\pm$3.7 & 51.7$\pm$4.3 \\
\midrule
\multirow{4}{*}{Reacher} & LeWM & 85.2$\pm$2.5 & 84.3$\pm$3.7 & 85.0$\pm$3.2 & 79.4$\pm$5.2 & 61.7$\pm$4.0 & 79.1$\pm$1.9 \\
 & SCALE & \textbf{85.8$\pm$3.9} & 84.8$\pm$3.9 & \textbf{85.8$\pm$3.7} & \textbf{80.2$\pm$3.1} & \textbf{63.0$\pm$3.2} & \textbf{79.9$\pm$2.5} \\
 & Aux & 84.9$\pm$3.5 & \textbf{85.9$\pm$2.9} & \textbf{85.8$\pm$3.4} & 80.1$\pm$3.5 & 62.5$\pm$4.5 & 79.8$\pm$1.8 \\
 & DINO-WM & 78.7$\pm$4.9 & 78.2$\pm$4.8 & 80.8$\pm$3.5 & 76.0$\pm$4.0 & 53.0$\pm$5.5 & 73.3$\pm$3.1 \\
\midrule
\multirow{4}{*}{Cube} & LeWM & 68.9$\pm$4.4 & 67.4$\pm$3.7 & 69.0$\pm$3.9 & 61.8$\pm$4.3 & 55.8$\pm$5.1 & 64.6$\pm$3.8 \\
 & SCALE & \textbf{76.8$\pm$3.5} & \textbf{76.6$\pm$3.8} & \textbf{75.1$\pm$2.8} & \textbf{65.7$\pm$2.9} & \textbf{59.3$\pm$4.2} & \textbf{70.7$\pm$2.9} \\
 & Aux & 74.0$\pm$3.7 & 73.0$\pm$4.7 & 73.0$\pm$3.5 & 64.7$\pm$3.4 & 57.9$\pm$3.8 & 68.5$\pm$3.3 \\
 & DINO-WM & 69.9$\pm$2.5 & 67.4$\pm$1.6 & 66.0$\pm$1.7 & 57.7$\pm$3.8 & 51.4$\pm$3.6 & 62.5$\pm$1.3 \\
\midrule
\multirow{4}{*}{Two-Room} & LeWM & 90.7$\pm$1.6 & 92.8$\pm$1.6 & 88.3$\pm$2.1 & 77.5$\pm$2.5 & 61.6$\pm$3.0 & 82.2$\pm$1.2 \\
 & SCALE & 99.9$\pm$0.3 & 99.7$\pm$0.6 & 97.9$\pm$1.2 & 90.3$\pm$3.2 & 76.7$\pm$2.5 & 92.9$\pm$1.0 \\
 & Aux & 92.4$\pm$2.0 & 93.4$\pm$1.6 & 88.4$\pm$2.0 & 78.6$\pm$2.4 & 62.5$\pm$2.8 & 83.1$\pm$1.1 \\
 & DINO-WM & \textbf{100.0$\pm$0.0} & \textbf{100.0$\pm$0.0} & \textbf{99.8$\pm$0.5} & \textbf{98.6$\pm$0.8} & \textbf{92.5$\pm$2.1} & \textbf{98.2$\pm$0.5} \\
\midrule
\multirow{4}{*}{PointMaze} & LeWM & 79.2$\pm$8.3 & 78.2$\pm$8.2 & 77.8$\pm$6.4 & 77.3$\pm$5.9 & 73.2$\pm$4.0 & 77.1$\pm$5.6 \\
 & SCALE & 84.4$\pm$6.3 & 82.8$\pm$6.7 & 80.0$\pm$8.0 & 77.7$\pm$7.3 & 73.5$\pm$6.5 & 79.7$\pm$6.3 \\
 & Aux & 85.0$\pm$5.8 & 83.4$\pm$5.9 & 83.5$\pm$6.9 & 81.5$\pm$4.7 & 76.4$\pm$6.5 & 82.0$\pm$5.1 \\
 & DINO-WM & \textbf{93.4$\pm$2.6} & \textbf{92.2$\pm$2.6} & \textbf{90.8$\pm$3.4} & \textbf{90.3$\pm$2.6} & \textbf{85.3$\pm$2.6} & \textbf{90.4$\pm$2.1} \\
\bottomrule
\end{tabular}

% CEM
\begin{tabular}{llcccccc}
\toprule
Task & Method & T1 & T2 & T3 & T4 & T5 & Overall \\
\midrule
\multirow{4}{*}{Push-T} & LeWM & 91.8$\pm$2.6 & 90.0$\pm$2.8 & 80.7$\pm$4.7 & 58.2$\pm$4.9 & 33.5$\pm$5.3 & 70.9$\pm$2.9 \\
 & SCALE & \textbf{95.1$\pm$2.7} & 92.2$\pm$2.9 & 84.4$\pm$3.3 & 62.7$\pm$6.2 & 33.7$\pm$5.6 & 73.6$\pm$2.8 \\
 & Aux & 94.5$\pm$2.3 & \textbf{92.6$\pm$1.9} & \textbf{84.9$\pm$4.2} & \textbf{64.5$\pm$5.4} & \textbf{34.4$\pm$5.5} & \textbf{74.2$\pm$2.9} \\
 & DINO-WM & 74.8$\pm$4.6 & 72.1$\pm$4.5 & 65.3$\pm$4.9 & 46.1$\pm$4.6 & 24.3$\pm$6.1 & 56.5$\pm$4.0 \\
\midrule
\multirow{4}{*}{Reacher} & LeWM & 81.8$\pm$3.1 & 79.1$\pm$3.9 & 73.4$\pm$4.1 & 65.5$\pm$4.8 & 56.4$\pm$6.0 & 71.2$\pm$2.5 \\
 & SCALE & 81.1$\pm$4.6 & \textbf{79.8$\pm$4.7} & 73.9$\pm$4.2 & 65.8$\pm$5.1 & \textbf{56.8$\pm$4.9} & 71.5$\pm$2.5 \\
 & Aux & \textbf{82.4$\pm$4.1} & \textbf{79.8$\pm$3.8} & \textbf{74.1$\pm$3.9} & \textbf{67.3$\pm$4.9} & 56.5$\pm$6.4 & \textbf{72.0$\pm$3.0} \\
 & DINO-WM & 77.2$\pm$2.8 & 74.8$\pm$5.4 & 70.8$\pm$5.9 & 61.8$\pm$3.4 & 50.3$\pm$7.3 & 67.0$\pm$3.0 \\
\midrule
\multirow{4}{*}{Cube} & LeWM & 67.8$\pm$4.6 & 66.4$\pm$3.9 & 60.4$\pm$4.2 & 56.5$\pm$3.6 & 54.5$\pm$3.9 & 61.1$\pm$3.4 \\
 & SCALE & \textbf{74.7$\pm$3.8} & \textbf{71.6$\pm$3.8} & \textbf{64.7$\pm$3.7} & 55.4$\pm$4.0 & \textbf{55.2$\pm$4.0} & \textbf{64.3$\pm$3.1} \\
 & Aux & 71.8$\pm$4.4 & 69.4$\pm$3.8 & 63.9$\pm$4.8 & \textbf{56.6$\pm$3.7} & 54.9$\pm$3.9 & 63.3$\pm$3.2 \\
 & DINO-WM & 73.4$\pm$3.1 & 66.9$\pm$2.3 & 58.4$\pm$4.4 & 51.6$\pm$4.5 & 53.9$\pm$3.8 & 60.9$\pm$2.9 \\
\midrule
\multirow{4}{*}{Two-Room} & LeWM & 87.3$\pm$2.7 & 86.3$\pm$3.0 & 82.3$\pm$3.7 & 71.8$\pm$2.5 & 58.0$\pm$3.0 & 77.1$\pm$1.5 \\
 & SCALE & 99.5$\pm$0.6 & 99.1$\pm$1.0 & 94.9$\pm$2.2 & 83.2$\pm$2.3 & 70.8$\pm$3.6 & 89.5$\pm$0.8 \\
 & Aux & 88.2$\pm$2.3 & 87.2$\pm$2.8 & 82.8$\pm$3.1 & 71.3$\pm$3.2 & 59.5$\pm$3.4 & 77.8$\pm$1.2 \\
 & DINO-WM & \textbf{100.0$\pm$0.0} & \textbf{99.8$\pm$0.4} & \textbf{98.3$\pm$1.2} & \textbf{94.1$\pm$2.2} & \textbf{85.8$\pm$3.7} & \textbf{95.6$\pm$1.0} \\
\midrule
\multirow{4}{*}{PointMaze} & LeWM & 84.5$\pm$4.1 & 84.0$\pm$4.6 & 83.8$\pm$5.0 & 81.4$\pm$4.7 & 75.8$\pm$5.4 & 81.9$\pm$3.3 \\
 & SCALE & 83.6$\pm$7.0 & 83.8$\pm$7.5 & 83.0$\pm$6.7 & 80.9$\pm$7.0 & 75.7$\pm$6.1 & 81.4$\pm$6.3 \\
 & Aux & 87.8$\pm$4.7 & 87.4$\pm$5.2 & 86.3$\pm$5.4 & 83.1$\pm$5.5 & 78.0$\pm$4.9 & 84.5$\pm$4.2 \\
 & DINO-WM & \textbf{98.2$\pm$1.2} & \textbf{98.3$\pm$1.4} & \textbf{97.2$\pm$1.3} & \textbf{92.4$\pm$3.0} & \textbf{87.2$\pm$2.9} & \textbf{94.7$\pm$1.4} \\
\bottomrule
\end{tabular}

```
