# Incremental Mining — Reflection

Data file: D:\project\lab_04\cleaned_comments.csv
Rows used: 3063

- Chunk sizes: [613, 613, 613, 612, 612]
- Correlation (unigrams, top 5):
          iphone     phone     apple      like      year
iphone  1.000000 -0.670251  0.345181  0.146185 -0.337933
phone  -0.670251  1.000000 -0.604556 -0.740075  0.724146
apple   0.345181 -0.604556  1.000000  0.856581 -0.870702
like    0.146185 -0.740075  0.856581  1.000000 -0.917424
year   -0.337933  0.724146 -0.870702 -0.917424  1.000000

- Correlation (pairs, top 5):
                every • year  max • pro  like • look  iphone • phone  iphone • like
every • year        1.000000   0.331063    -0.861136        0.796539      -0.856377
max • pro           0.331063   1.000000    -0.312774       -0.150649      -0.257629
like • look        -0.861136  -0.312774     1.000000       -0.585116       0.711324
iphone • phone      0.796539  -0.150649    -0.585116        1.000000      -0.508974
iphone • like      -0.856377  -0.257629     0.711324       -0.508974       1.000000


Guidance:
1) Describe visible rises/falls in the line plots (figures/*_lines_over_chunks.png).
2) Pick 3–5 patterns from the correlation inputs and write 2–3 sentences each.
3) Acknowledge limitation: row chunking approximates time unless truly timestamp-sorted.
