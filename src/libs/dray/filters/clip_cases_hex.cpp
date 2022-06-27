
int numClipCasesHex = 256;

int numClipShapesHex[256] = {
  13, 18, 0, 20, 0, 28, 0, 23,  // cases 0 - 7
  0, 0, 0, 0, 0, 0, 0, 21,  // cases 8 - 15
  0, 0, 0, 0, 23, 30, 0, 25,  // cases 16 - 23
  0, 0, 39, 25, 0, 0, 34, 23,  // cases 24 - 31
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 32 - 39
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 40 - 47
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 48 - 55
  0, 0, 0, 0, 33, 28, 0, 20,  // cases 56 - 63
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 64 - 71
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 72 - 79
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 80 - 87
  0, 0, 51, 37, 0, 0, 0, 26,  // cases 88 - 95
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 96 - 103
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 104 - 111
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 112 - 119
  0, 0, 0, 0, 0, 23, 0, 18,  // cases 120 - 127
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 128 - 135
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 136 - 143
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 144 - 151
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 152 - 159
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 160 - 167
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 168 - 175
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 176 - 183
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 184 - 191
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 192 - 199
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 200 - 207
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 208 - 215
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 216 - 223
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 224 - 231
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 232 - 239
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 240 - 247
  0, 0, 0, 0, 0, 0, 0, 13 // cases 248 - 255
};

int clipShapesHex[256] = {
  0, 96, 227, 227, 371, 371, 567, 567,  // cases 0 - 7
  731, 731, 731, 731, 731, 731, 731, 731,  // cases 8 - 15
  879, 879, 879, 879, 879, 1043, 1256, 1256,  // cases 16 - 23
  1435, 1435, 1435, 1724, 1903, 1903, 1903, 2152,  // cases 24 - 31
  2316, 2316, 2316, 2316, 2316, 2316, 2316, 2316,  // cases 32 - 39
  2316, 2316, 2316, 2316, 2316, 2316, 2316, 2316,  // cases 40 - 47
  2316, 2316, 2316, 2316, 2316, 2316, 2316, 2316,  // cases 48 - 55
  2316, 2316, 2316, 2316, 2316, 2557, 2761, 2761,  // cases 56 - 63
  2905, 2905, 2905, 2905, 2905, 2905, 2905, 2905,  // cases 64 - 71
  2905, 2905, 2905, 2905, 2905, 2905, 2905, 2905,  // cases 72 - 79
  2905, 2905, 2905, 2905, 2905, 2905, 2905, 2905,  // cases 80 - 87
  2905, 2905, 2905, 3288, 3562, 3562, 3562, 3562,  // cases 88 - 95
  3752, 3752, 3752, 3752, 3752, 3752, 3752, 3752,  // cases 96 - 103
  3752, 3752, 3752, 3752, 3752, 3752, 3752, 3752,  // cases 104 - 111
  3752, 3752, 3752, 3752, 3752, 3752, 3752, 3752,  // cases 112 - 119
  3752, 3752, 3752, 3752, 3752, 3752, 3917, 3917,  // cases 120 - 127
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 128 - 135
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 136 - 143
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 144 - 151
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 152 - 159
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 160 - 167
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 168 - 175
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 176 - 183
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 184 - 191
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 192 - 199
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 200 - 207
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 208 - 215
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 216 - 223
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 224 - 231
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 232 - 239
  0, 0, 0, 0, 0, 0, 0, 0,  // cases 240 - 247
  0, 0, 0, 0, 0, 0, 0, 4048 // cases 248 - 255
};

unsigned char clipShapesHex[] = {
 // Case #0: Unique case #1
  ST_PNT, 0, COLOR0, 8, P0, P1, P2, P3, P4, P5, P6, P7,
  ST_TET, COLOR0, P3, P4, P0, N0,
  ST_TET, COLOR0, P3, P7, P4, N0,
  ST_TET, COLOR0, P2, P1, P5, N0,
  ST_TET, COLOR0, P2, P5, P6, N0,
  ST_TET, COLOR0, P3, P0, P1, N0,
  ST_TET, COLOR0, P3, P1, P2, N0,
  ST_TET, COLOR0, P7, P5, P4, N0,
  ST_TET, COLOR0, P7, P6, P5, N0,
  ST_TET, COLOR0, P3, P6, P7, N0,
  ST_TET, COLOR0, P3, P2, P6, N0,
  ST_TET, COLOR0, P0, P4, P5, N0,
  ST_TET, COLOR0, P0, P5, P1, N0,
 // Case #1: Unique case #2
  ST_PNT, 0, NOCOLOR, 7, P1, P2, P3, P4, P5, P6, P7,
  ST_TET, COLOR0, P7, ED, P3, N0,
  ST_TET, COLOR0, P7, EI, ED, N0,
  ST_TET, COLOR0, P7, P4, EI, N0,
  ST_TET, COLOR0, P2, P5, P6, N0,
  ST_TET, COLOR0, P2, P1, P5, N0,
  ST_TET, COLOR0, P2, P3, ED, N0,
  ST_TET, COLOR0, P2, ED, EA, N0,
  ST_TET, COLOR0, P2, EA, P1, N0,
  ST_TET, COLOR0, P7, P5, P4, N0,
  ST_TET, COLOR0, P7, P6, P5, N0,
  ST_TET, COLOR0, P3, P6, P7, N0,
  ST_TET, COLOR0, P3, P2, P6, N0,
  ST_TET, COLOR0, P5, P1, EA, N0,
  ST_TET, COLOR0, P5, EA, EI, N0,
  ST_TET, COLOR0, P5, EI, P4, N0,
  ST_TET, COLOR0, ED, EI, EA, N0,
  ST_TET, COLOR1, ED, P0, EA, EI,
 // Case #2: (cloned #1)
 // Case #3: Unique case #3
  ST_PNT, 0, NOCOLOR, 6, P2, P3, P4, P5, P6, P7,
  ST_TET, COLOR0, P7, ED, P3, N0,
  ST_TET, COLOR0, P7, EI, ED, N0,
  ST_TET, COLOR0, P7, P4, EI, N0,
  ST_TET, COLOR0, P6, P2, EB, N0,
  ST_TET, COLOR0, P6, EB, EJ, N0,
  ST_TET, COLOR0, P6, EJ, P5, N0,
  ST_TET, COLOR0, P3, ED, EB, N0,
  ST_TET, COLOR0, P3, EB, P2, N0,
  ST_TET, COLOR0, P7, P5, P4, N0,
  ST_TET, COLOR0, P7, P6, P5, N0,
  ST_TET, COLOR0, P3, P6, P7, N0,
  ST_TET, COLOR0, P3, P2, P6, N0,
  ST_TET, COLOR0, ED, EI, EJ, N0,
  ST_TET, COLOR0, ED, EJ, EB, N0,
  ST_TET, COLOR0, EI, P4, P5, N0,
  ST_TET, COLOR0, EI, P5, EJ, N0,
  ST_TET, COLOR1, ED, P0, EJ, EI,
  ST_TET, COLOR1, ED, P0, P1, EJ,
  ST_TET, COLOR1, ED, P1, EB, EJ,
 // Case #4: (cloned #1)
 // Case #5: Unique case #4
  ST_PNT, 0, NOCOLOR, 2, EI, EL, 
  ST_PNT, 1, NOCOLOR, 4, EA, EB, EC, ED
  ST_TET, COLOR0, P7, ED, P3, N0,
  ST_TET, COLOR0, P7, EI, ED, N0,
  ST_TET, COLOR0, P7, P4, EI, N0,
  ST_TET, COLOR0, P5, P6, EL, N0,
  ST_TET, COLOR0, P5, EL, EB, N0,
  ST_TET, COLOR0, P5, EB, P1, N0,
  ST_TET, COLOR0, P3, ED, EC, N0,
  ST_TET, COLOR0, EA, P1, EB, N0,
  ST_TET, COLOR0, P7, P5, P4, N0,
  ST_TET, COLOR0, P7, P6, P5, N0,
  ST_TET, COLOR0, P7, P3, EC, N0,
  ST_TET, COLOR0, P7, EC, EL, N0,
  ST_TET, COLOR0, P7, EL, P6, N0,
  ST_TET, COLOR0, P5, P1, EA, N0,
  ST_TET, COLOR0, P5, EA, EI, N0,
  ST_TET, COLOR0, P5, EI, P4, N0,
  ST_TET, COLOR1, ED, EI, P0, N0,
  ST_TET, COLOR1, EI, EA, P0, N0,
  ST_TET, COLOR1, EC, P2, EL, N0,
  ST_TET, COLOR1, P2, EB, EL, N0,
  ST_TET, COLOR1, N1, P0, EA, N0,
  ST_TET, COLOR1, N1, EA, EB, N0,
  ST_TET, COLOR1, N1, EB, P2, N0,
  ST_TET, COLOR1, N1, P2, EC, N0,
  ST_TET, COLOR1, N1, EC, ED, N0,
  ST_TET, COLOR1, N1, ED, P0, N0
 // Case #6: (cloned #3)
 // Case #7: Unique case #5
  ST_PNT, 0, NOCOLOR, 5, EI, EJ, ED, EC, EL, 
  ST_TET, COLOR0, P7, ED, P3, N0,
  ST_TET, COLOR0, P7, EI, ED, N0,
  ST_TET, COLOR0, P7, P4, EI, N0,
  ST_TET, COLOR0, EL, P5, P6, N0,
  ST_TET, COLOR0, EL, EJ, P5, N0,
  ST_TET, COLOR0, P3, ED, EC, N0,
  ST_TET, COLOR0, P7, P5, P4, N0,
  ST_TET, COLOR0, P7, P6, P5, N0,
  ST_TET, COLOR0, P7, P3, EC, N0,
  ST_TET, COLOR0, P7, EC, EL, N0,
  ST_TET, COLOR0, P7, EL, P6, N0,
  ST_TET, COLOR0, EI, P4, P5, N0,
  ST_TET, COLOR0, EI, P5, EJ, N0,
  ST_TET, COLOR1, ED, EI, P0, N0,
  ST_TET, COLOR1, P2, EJ, EL, N0,
  ST_TET, COLOR1, P2, P1, EJ, N0,
  ST_TET, COLOR1, P1, ED, P0, N0,
  ST_TET, COLOR1, P1, EC, ED, N0,
  ST_TET, COLOR1, P1, P2, EC, N0,
  ST_TET, COLOR1, EC, P2, EL, N0,
  ST_TET, COLOR1, P0, EI, EJ, N0,
  ST_TET, COLOR1, P0, EJ, P1, N0,
 // Case #8: (cloned #1)
 // Case #9: (cloned #3)
 // Case #10: (cloned #5)
 // Case #11: (cloned #7)
 // Case #12: (cloned #3)
 // Case #13: (cloned #7)
 // Case #14: (cloned #7)
 // Case #15: Unique case #6
  ST_PNT, 0, NOCOLOR, 4, EI, EJ, EL, EK
  ST_TET, COLOR0, EK, P7, P4, N0,
  ST_TET, COLOR0, EK, P4, EI, N0,
  ST_TET, COLOR0, EL, EJ, P5, N0,
  ST_TET, COLOR0, EL, P5, P6, N0,
  ST_TET, COLOR0, P7, P6, P5, N0,
  ST_TET, COLOR0, P7, P5, P4, N0,
  ST_TET, COLOR0, EK, P6, P7, N0,
  ST_TET, COLOR0, EK, EL, P6, N0,
  ST_TET, COLOR0, EI, P5, EJ, N0,
  ST_TET, COLOR0, EI, P4, P5, N0,
  ST_TET, COLOR1, P3, EK, EI, N0,
  ST_TET, COLOR1, P3, EI, P0, N0,
  ST_TET, COLOR1, P2, EJ, EL, N0,
  ST_TET, COLOR1, P2, P1, EJ, N0,
  ST_TET, COLOR1, P3, P1, P2, N0,
  ST_TET, COLOR1, P3, P0, P1, N0,
  ST_TET, COLOR1, P3, P2, EL, N0,
  ST_TET, COLOR1, P3, EL, EK, N0,
  ST_TET, COLOR1, P0, EJ, P1, N0,
  ST_TET, COLOR1, P0, EI, EJ, N0,
 // Case #16: (cloned #1)
 // Case #17: (cloned #3)
 // Case #18: (cloned #5)
 // Case #19: (cloned #7)
 // Case #20: Unique case #7
  ST_PNT, 0, NOCOLOR, 6, P0, P1, P3, P5, P6, P7
  ST_TET, COLOR0, P3, P7, EH, N0,
  ST_TET, COLOR0, P3, EH, EI, N0,
  ST_TET, COLOR0, P3, EI, P0, N0,
  ST_TET, COLOR0, P5, P6, EL, N0,
  ST_TET, COLOR0, P5, EL, EB, N0,
  ST_TET, COLOR0, P5, EB, P1, N0,
  ST_TET, COLOR0, P0, P1, EB, N0,
  ST_TET, COLOR0, P0, EB, EC, N0,
  ST_TET, COLOR0, P0, EC, P3, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, P7, P3, EC, N0,
  ST_TET, COLOR0, P7, EC, EL, N0,
  ST_TET, COLOR0, P7, EL, P6, N0,
  ST_TET, COLOR0, P1, EE, P5, N0,
  ST_TET, COLOR0, P1, EI, EE, N0,
  ST_TET, COLOR0, P1, P0, EI, N0,
  ST_TET, COLOR0, EC, EB, EL, N0,
  ST_TET, COLOR0, EH, EE, EI, N0,
  ST_TET, COLOR1, EI, EE, EH, P4,
  ST_TET, COLOR1, P2, EC, EB, EL,
 // Case #21: Unique case #8
  ST_PNT, 0, NOCOLOR, 4, EE, EH, EL, EL, 
  ST_PNT, 1, NOCOLOR, 4, EA, EB, EC, ED
  ST_TET, COLOR0, P3, P7, EH, N0,
  ST_TET, COLOR0, P3, EH, ED, N0,
  ST_TET, COLOR0, P5, P6, EL, N0,
  ST_TET, COLOR0, P5, EL, EB, N0,
  ST_TET, COLOR0, P5, EB, P1, N0,
  ST_TET, COLOR0, P3, ED, EC, N0,
  ST_TET, COLOR0, EA, P1, EB, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, P7, P3, EC, N0,
  ST_TET, COLOR0, P7, EC, EL, N0,
  ST_TET, COLOR0, P7, EL, P6, N0,
  ST_TET, COLOR0, EA, EE, P5, N0,
  ST_TET, COLOR0, EA, P5, P1, N0,
  ST_TET, COLOR1, ED, EH, P4, N0,
  ST_TET, COLOR1, ED, P4, P0, N0,
  ST_TET, COLOR1, P2, EB, EL, N0,
  ST_TET, COLOR1, N1, P0, EA, N0,
  ST_TET, COLOR1, N1, EA, EB, N0,
  ST_TET, COLOR1, N1, EB, P2, N0,
  ST_TET, COLOR1, N1, P2, EC, N0,
  ST_TET, COLOR1, N1, EC, ED, N0,
  ST_TET, COLOR1, N1, ED, P0, N0,
  ST_TET, COLOR1, P4, EH, EE, N0,
  ST_TET, COLOR1, EC, P2, EL, N0,
  ST_TET, COLOR1, P0, P4, EE, N0,
  ST_TET, COLOR1, P0, EE, EA, N0,
 // Case #22: (cloned #21)
 // Case #23: Unique case #9
  ST_PNT, 0, NOCOLOR, 6, ED, EC, EL, EJ, EE, EH, 
  ST_TET, COLOR0, P3, P7, EH, N0,
  ST_TET, COLOR0, P3, EH, ED, N0,
  ST_TET, COLOR0, EL, EJ, P5, N0,
  ST_TET, COLOR0, EL, P5, P6, N0,
  ST_TET, COLOR0, P3, ED, EC, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, P7, P3, EC, N0,
  ST_TET, COLOR0, P7, EC, EL, N0,
  ST_TET, COLOR0, P7, EL, P6, N0,
  ST_TET, COLOR0, P5, EJ, EE, N0,
  ST_TET, COLOR1, ED, EH, P4, N0,
  ST_TET, COLOR1, ED, P4, P0, N0,
  ST_TET, COLOR1, P2, P1, EJ, N0,
  ST_TET, COLOR1, P2, EJ, EL, N0,
  ST_TET, COLOR1, P1, P2, EC, N0,
  ST_TET, COLOR1, P1, EC, ED, N0,
  ST_TET, COLOR1, P1, ED, P0, N0,
  ST_TET, COLOR1, P4, EH, EE, N0,
  ST_TET, COLOR1, P2, EL, EC, N0,
  ST_TET, COLOR1, P0, P4, EE, N0,
  ST_TET, COLOR1, P0, EE, EJ, N0,
  ST_TET, COLOR1, P0, EJ, P1, N0,
 // Case #24: (cloned #5)
 // Case #25: (cloned #7)
 // Case #26: Unique case #10
  ST_PNT, 0, NOCOLOR, 6, EB, EC, EK, EH, EE, EJ,
  ST_PNT, 1, NOCOLOR, 6, P3, ED, EI, P4, EH, EK,
  ST_PNT, 2, NOCOLOR, 6, P3, ED, EA, P1, EB, EC,
  ST_PNT, 3, NOCOLOR, 6, EA, P1, EJ, EE, P4, EI,
  ST_TET, COLOR0, P0, EA, ED, EI,
  ST_TET, COLOR0, P7, EH, EK, N0,
  ST_TET, COLOR0, P6, P2, EB, N0,
  ST_TET, COLOR0, P6, EB, EJ, N0,
  ST_TET, COLOR0, P6, EJ, P5, N0,
  ST_TET, COLOR0, P2, EC, EB, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, P6, P7, EK, N0,
  ST_TET, COLOR0, P6, EK, EC, N0,
  ST_TET, COLOR0, P6, EC, P2, N0,
  ST_TET, COLOR0, P5, EJ, EE, N0,
  ST_TET, COLOR1, N1, P3, EK, N0,
  ST_TET, COLOR1, N1, EK, EH, N0,
  ST_TET, COLOR1, N1, EH, P4, N0,
  ST_TET, COLOR1, N1, P4, EI, N0,
  ST_TET, COLOR1, N1, EI, ED, N0,
  ST_TET, COLOR1, N1, ED, P3, N0,
  ST_TET, COLOR1, P1, EJ, EB, N0,
  ST_TET, COLOR1, N2, EA, P1, N0, 
  ST_TET, COLOR1, N2, P1, EB, N0, 
  ST_TET, COLOR1, N2, EB, EC, N0, 
  ST_TET, COLOR1, N2, EC, P3, N0, 
  ST_TET, COLOR1, N2, P3, ED, N0, 
  ST_TET, COLOR1, N2, ED, EA, N0, 
  ST_TET, COLOR1, P4, EH, EE, N0,
  ST_TET, COLOR1, P3, EC, EK, N0,
  ST_TET, COLOR1, N3, P4, EE, N0, 
  ST_TET, COLOR1, N3, EE, EJ, N0, 
  ST_TET, COLOR1, N3, EJ, P1, N0, 
  ST_TET, COLOR1, N3, P1, EA, N0, 
  ST_TET, COLOR1, N3, EA, EI, N0, 
  ST_TET, COLOR1, N3, EI, P4, N0, 
  ST_TET, COLOR1, ED, EI, EA, N0,
 // Case #27: Unique case #11
  ST_PNT, 0, NOCOLOR, 6, EC, EB, EJ, EE, EH, EK,
  ST_TET, COLOR0, P7, EH, EK, N0,
  ST_TET, COLOR0, P6, P2, EB, N0,
  ST_TET, COLOR0, P6, EB, EJ, N0,
  ST_TET, COLOR0, P6, EJ, P5, N0,
  ST_TET, COLOR0, P2, EC, EB, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, P6, P7, EK, N0,
  ST_TET, COLOR0, P6, EK, EC, N0,
  ST_TET, COLOR0, P6, EC, P2, N0,
  ST_TET, COLOR0, P5, EJ, EE, N0,
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, P1, EJ, EB, N0,
  ST_TET, COLOR1, P0, P1, EB, N0,
  ST_TET, COLOR1, P0, EB, EC, N0,
  ST_TET, COLOR1, P0, EC, P3, N0,
  ST_TET, COLOR1, P4, EH, EE, N0,
  ST_TET, COLOR1, P3, EC, EK, N0,
  ST_TET, COLOR1, P0, P4, EE, N0,
  ST_TET, COLOR1, P0, EE, EJ, N0,
  ST_TET, COLOR1, P0, EJ, P1, N0,
 // Case #28: (cloned #21)
 // Case #29: (cloned #23)
 // Case #30: Unique case #12
  ST_PNT, 0, NOCOLOR, 5, EL, EJ, EK, EH, EE,
  ST_PNT, 1, NOCOLOR, 6, P3, ED, EI, P4, EH, EK,
  ST_PNT, 2, NOCOLOR, 6, EA, P1, EJ, EE, P4, EI,
  ST_TET, COLOR0, P0, EA, ED, EI,
  ST_TET, COLOR0, P7, EH, EK, N0,
  ST_TET, COLOR0, EL, EJ, P5, N0,
  ST_TET, COLOR0, EL, P5, P6, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, EK, EL, P6, N0,
  ST_TET, COLOR0, EK, P6, P7, N0,
  ST_TET, COLOR0, P5, EJ, EE, N0,
  ST_TET, COLOR1, N1, ED, P3, N0,
  ST_TET, COLOR1, N1, P3, EK, N0,
  ST_TET, COLOR1, N1, EK, EH, N0,
  ST_TET, COLOR1, N1, EH, P4, N0,
  ST_TET, COLOR1, N1, P4, EI, N0,
  ST_TET, COLOR1, N1, EI, ED, N0,
  ST_TET, COLOR1, P2, P1, EJ, N0,
  ST_TET, COLOR1, P2, EJ, EL, N0,
  ST_TET, COLOR1, P2, P3, ED, N0,
  ST_TET, COLOR1, P2, ED, EA, N0,
  ST_TET, COLOR1, P2, EA, P1, N0,
  ST_TET, COLOR1, P4, EH, EE, N0,
  ST_TET, COLOR1, P3, P2, EL, N0,
  ST_TET, COLOR1, P3, EL, EK, N0,
  ST_TET, COLOR1, N2, P4, EE, N0,
  ST_TET, COLOR1, N2, EE, EJ, N0,
  ST_TET, COLOR1, N2, EJ, P1, N0,
  ST_TET, COLOR1, N2, P1, EA, N0,
  ST_TET, COLOR1, N2, EA, EI, N0,
  ST_TET, COLOR1, N2, EI, P4, N0,
  ST_TET, COLOR1, ED, EI, EA, N0,
 // Case #31: Unique case #13
  ST_PNT, 0, NOCOLOR, 5, EJ, EL, EK, EE, EH, 
  ST_TET, COLOR0, P7, EH, EK, N0,
  ST_TET, COLOR0, EL, EJ, P5, N0,
  ST_TET, COLOR0, EL, P5, P6, N0,
  ST_TET, COLOR0, P6, P5, EE, N0,
  ST_TET, COLOR0, P6, EE, EH, N0,
  ST_TET, COLOR0, P6, EH, P7, N0,
  ST_TET, COLOR0, EK, EL, P6, N0,
  ST_TET, COLOR0, EK, P6, P7, N0,
  ST_TET, COLOR0, P5, EJ, EE, N0,
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, P2, P1, EJ, N0,
  ST_TET, COLOR1, P2, EJ, EL, N0,
  ST_TET, COLOR1, P3, P0, P1, N0,
  ST_TET, COLOR1, P3, P1, P2, N0,
  ST_TET, COLOR1, P4, EH, EE, N0,
  ST_TET, COLOR1, P3, P2, EL, N0,
  ST_TET, COLOR1, P3, EL, EK, N0,
  ST_TET, COLOR1, P0, P4, EE, N0,
  ST_TET, COLOR1, P0, EE, EJ, N0,
  ST_TET, COLOR1, P0, EJ, P1, N0,
 // Case #32: (cloned #1)
 // Case #33: (cloned #5)
 // Case #34: (cloned #3)
 // Case #35: (cloned #7)
 // Case #36: (cloned #5)
 // Case #37: (cloned #26)
 // Case #38: (cloned #7)
 // Case #39: (cloned #27)
 // Case #40: (cloned #20)
 // Case #41: (cloned #21)
 // Case #42: (cloned #21)
 // Case #43: (cloned #23)
 // Case #44: (cloned #21)
 // Case #45: (cloned #30)
 // Case #46: (cloned #23)
 // Case #47: (cloned #31)
 // Case #48: (cloned #3)
 // Case #49: (cloned #7)
 // Case #50: (cloned #7)
 // Case #51: (cloned #15)
 // Case #52: (cloned #21)
 // Case #53: (cloned #30)
 // Case #54: (cloned #23)
 // Case #55: (cloned #31)
 // Case #56: (cloned #21)
 // Case #57: (cloned #23)
 // Case #58: (cloned #30)
 // Case #59: (cloned #31)
 // Case #60: Unique case #14
  ST_PNT, 0, NOCOLOR, 4, P2, P3, P4, P5,
  ST_PNT, 1, NOCOLOR, 6, ED, P3, EK, EH, P4, EI,
  ST_PNT, 2, NOCOLOR, 6, EB, P2, EL, EF, P5, EJ,
  ST_TET, COLOR0, ED, EI, P0, EJ,
  ST_TET, COLOR0, ED, P0, P1, EJ,
  ST_TET, COLOR0, ED, P1, EB, EJ,
  ST_TET, COLOR0, P7, EH, EK, EF,
  ST_TET, COLOR0, EK, EL, P6, EF,
  ST_TET, COLOR0, EK, P6, P7, EF,
  ST_TET, COLOR1, N1, ED, P3, N0,
  ST_TET, COLOR1, N1, P3, EK, N0,
  ST_TET, COLOR1, N1, EK, EH, N0,
  ST_TET, COLOR1, N1, EH, P4, N0,
  ST_TET, COLOR1, N1, P4, EI, N0,
  ST_TET, COLOR1, N1, EI, ED, N0,
  ST_TET, COLOR1, N2, P2, EB, N0,
  ST_TET, COLOR1, N2, EB, EJ, N0,
  ST_TET, COLOR1, N2, EJ, P5, N0,
  ST_TET, COLOR1, N2, P5, EF, N0,
  ST_TET, COLOR1, N2, EF, EL, N0,
  ST_TET, COLOR1, N2, EL, P2, N0,
  ST_TET, COLOR1, P3, ED, EB, N0,
  ST_TET, COLOR1, P3, EB, P2, N0,
  ST_TET, COLOR1, EH, EF, P5, N0,
  ST_TET, COLOR1, EH, P5, P4, N0,
  ST_TET, COLOR1, P3, P2, EL, N0,
  ST_TET, COLOR1, P3, EL, EK, N0,
  ST_TET, COLOR1, EI, P4, P5, N0,
  ST_TET, COLOR1, EI, P5, EJ, N0,
  ST_TET, COLOR1, ED, EI, EJ, N0,
  ST_TET, COLOR1, ED, EJ, EB, N0,
  ST_TET, COLOR1, EK, EL, EF, N0,
  ST_TET, COLOR1, EK, EF, EH, N0,
 // Case #61: Unique case #15
  ST_PNT, 0, NOCOLOR, 6, P0, P2, P3, P4, EF, EH, 
  ST_PNT, 1, NOCOLOR, 6, EB, P2, EL, EF, P5, EJ,
  ST_TET, COLOR0, P7, EH, EK, EF,
  ST_TET, COLOR0, EK, EL, P6, EF,
  ST_TET, COLOR0, EK, P6, P7, EF,
  ST_TET, COLOR0, EA, P1, EB, EJ,
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, N1, EB, EJ, N0,
  ST_TET, COLOR1, N1, EJ, P5, N0,
  ST_TET, COLOR1, N1, P5, EF, N0,
  ST_TET, COLOR1, N1, EF, EL, N0,
  ST_TET, COLOR1, N1, EL, P2, N0,
  ST_TET, COLOR1, N1, P2, EB, N0,
  ST_TET, COLOR1, P3, P0, EA, N0,
  ST_TET, COLOR1, P3, EA, EB, N0,
  ST_TET, COLOR1, P3, EB, P2, N0,
  ST_TET, COLOR1, EH, EF, P5, N0,
  ST_TET, COLOR1, EH, P5, P4, N0,
  ST_TET, COLOR1, P3, P2, EL, N0,
  ST_TET, COLOR1, P3, EL, EK, N0,
  ST_TET, COLOR1, P4, P5, EJ, N0,
  ST_TET, COLOR1, P4, EJ, EA, N0,
  ST_TET, COLOR1, P4, EA, P0, N0,
  ST_TET, COLOR1, EA, EJ, EB, N0,
  ST_TET, COLOR1, EK, EF, EH, N0,
  ST_TET, COLOR1, EK, EL, EF, N0,
 // Case #62: (cloned #61)
 // Case #63: Unique case #16
  ST_PNT, 0, NOCOLOR, 6, P0, P1, P2, P3, P4, P5,
  ST_TET, COLOR0, P7, EH, EK, EF,
  ST_TET, COLOR0, EK, EL, P6, EF,
  ST_TET, COLOR0, EK, P6, P7, EF,
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, P1, P5, EF, N0,
  ST_TET, COLOR1, P1, EF, EL, N0,
  ST_TET, COLOR1, P1, EL, P2, N0,
  ST_TET, COLOR1, P3, P0, P1, N0,
  ST_TET, COLOR1, P3, P1, P2, N0,
  ST_TET, COLOR1, EH, EF, P5, N0,
  ST_TET, COLOR1, EH, P5, P4, N0,
  ST_TET, COLOR1, P3, P2, EL, N0,
  ST_TET, COLOR1, P3, EL, EK, N0,
  ST_TET, COLOR1, P0, P4, P5, N0,
  ST_TET, COLOR1, P0, P5, P1, N0,
  ST_TET, COLOR1, EK, EF, EH, N0,
  ST_TET, COLOR1, EK, EL, EF, N0,
 // Case #64: (cloned #1)
 // Case #65: (cloned #20)
 // Case #66: (cloned #5)
 // Case #67: (cloned #21)
 // Case #68: (cloned #3)
 // Case #69: (cloned #21)
 // Case #70: (cloned #7)
 // Case #71: (cloned #23)
 // Case #72: (cloned #5)
 // Case #73: (cloned #21)
 // Case #74: (cloned #26)
 // Case #75: (cloned #30)
 // Case #76: (cloned #7)
 // Case #77: (cloned #23)
 // Case #78: (cloned #27)
 // Case #79: (cloned #31)
 // Case #80: (cloned #5)
 // Case #81: (cloned #21)
 // Case #82: (cloned #26)
 // Case #83: (cloned #30)
 // Case #84: (cloned #21)
 // Case #85: (cloned #60)
 // Case #86: (cloned #30)
 // Case #87: (cloned #61)
 // Case #88: (cloned #26)
 // Case #89: (cloned #30)
 // Case #90: Unique case #17
  ST_TET, COLOR0, ED, P0, EA, EI,
  ST_TET, COLOR0, P2, EC, EB, EL,
  ST_TET, COLOR0, P7, EH, EK, EG,
  ST_TET, COLOR0, EJ, P5, EF, EE,
  ST_PNT, 0, COLOR1, 4, P3, P1, P4, P6,
  ST_PNT, 1, COLOR1, 6, ED, P3, EK, EH, P4, EI,
  ST_PNT, 2, COLOR1, 6, P1, EB, EL, P6, EF, EJ,
  ST_PNT, 3, COLOR1, 6, EA, P1, EB, EC, P3, ED,
  ST_PNT, 4, COLOR1, 6, P4, EE, EF, P6, EG, EH,
  ST_PNT, 5, COLOR1, 6, P3, EC, EL, P6, EG, EK,
  ST_PNT, 6, COLOR1, 6, EA, P1, EJ, EE, P4, EI,
  ST_TET, COLOR1, N1, EI, ED, N0,
  ST_TET, COLOR1, N1, ED, P3, N0,
  ST_TET, COLOR1, N1, P3, EK, N0,
  ST_TET, COLOR1, N1, EK, EH, N0,
  ST_TET, COLOR1, N1, EH, P4, N0,
  ST_TET, COLOR1, N1, P4, EI, N0,
  ST_TET, COLOR1, N2, EB, P1, N0,
  ST_TET, COLOR1, N2, P1, EJ, N0,
  ST_TET, COLOR1, N2, EJ, EF, N0,
  ST_TET, COLOR1, N2, EF, P6, N0,
  ST_TET, COLOR1, N2, P6, EL, N0,
  ST_TET, COLOR1, N2, EL, EB, N0,
  ST_TET, COLOR1, N3, EA, P1, N0,
  ST_TET, COLOR1, N3, P1, EB, N0,
  ST_TET, COLOR1, N3, EB, EC, N0,
  ST_TET, COLOR1, N3, EC, P3, N0,
  ST_TET, COLOR1, N3, P3, ED, N0,
  ST_TET, COLOR1, N3, ED, EA, N0,
  ST_TET, COLOR1, N4, EG, P6, N0,
  ST_TET, COLOR1, N4, P6, EF, N0,
  ST_TET, COLOR1, N4, EF, EE, N0,
  ST_TET, COLOR1, N4, EE, P4, N0,
  ST_TET, COLOR1, N4, P4, EH, N0,
  ST_TET, COLOR1, N4, EH, EG, N0,
  ST_TET, COLOR1, N5, P3, EC, N0,
  ST_TET, COLOR1, N5, EC, EL, N0,
  ST_TET, COLOR1, N5, EL, P6, N0,
  ST_TET, COLOR1, N5, P6, EG, N0,
  ST_TET, COLOR1, N5, EG, EK, N0,
  ST_TET, COLOR1, N5, EK, P3, N0,
  ST_TET, COLOR1, N6, P4, EE, N0,
  ST_TET, COLOR1, N6, EE, EJ, N0,
  ST_TET, COLOR1, N6, EJ, P1, N0,
  ST_TET, COLOR1, N6, P1, EA, N0,
  ST_TET, COLOR1, N6, EA, EI, N0,
  ST_TET, COLOR1, N6, EI, P4, N0,
  ST_TET, COLOR1, ED, EI, EA, N0,
  ST_TET, COLOR1, EC, EB, EL, N0,
  ST_TET, COLOR1, EH, EK, EG, N0,
  ST_TET, COLOR1, EF, EJ, EE, N0,
 // Case #91: Unique case #18
  ST_TET, COLOR0, EK, EH, EG, P7,
  ST_TET, COLOR0, EE, EJ, EF, P5,
  ST_TET, COLOR0, P2, EC, EB, EL,
  ST_PNT, 0, COLOR1, 5, P0, P1, P3, P4, P6,
  ST_PNT, 1, COLOR1, 6, P1, EB, EL, P6, EF, EJ,
  ST_PNT, 2, COLOR1, 6, P4, EE, EF, P6, EG, EH,
  ST_PNT, 3, COLOR1, 6, P3, EC, EL, P6, EG, EK,
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, N1, P1, EJ, N0,
  ST_TET, COLOR1, N1, EJ, EF, N0,
  ST_TET, COLOR1, N1, EF, P6, N0,
  ST_TET, COLOR1, N1, P6, EL, N0,
  ST_TET, COLOR1, N1, EL, EB, N0,
  ST_TET, COLOR1, N1, EB, P1, N0,
  ST_TET, COLOR1, P0, P1, EB, N0,
  ST_TET, COLOR1, P0, EB, EC, N0,
  ST_TET, COLOR1, P0, EC, P3, N0,
  ST_TET, COLOR1, N2, EG, P6, N0,
  ST_TET, COLOR1, N2, P6, EF, N0,
  ST_TET, COLOR1, N2, EF, EE, N0,
  ST_TET, COLOR1, N2, EE, P4, N0,
  ST_TET, COLOR1, N2, P4, EH, N0,
  ST_TET, COLOR1, N2, EH, EG, N0,
  ST_TET, COLOR1, N3, P3, EC, N0,
  ST_TET, COLOR1, N3, EC, EL, N0,
  ST_TET, COLOR1, N3, EL, P6, N0,
  ST_TET, COLOR1, N3, P6, EG, N0,
  ST_TET, COLOR1, N3, EG, EK, N0,
  ST_TET, COLOR1, N3, EK, P3, N0,
  ST_TET, COLOR1, P0, P4, EE, N0, 
  ST_TET, COLOR1, P0, EE, EJ, N0, 
  ST_TET, COLOR1, P0, EJ, P1, N0,
  ST_TET, COLOR1, EG, EH, EK, N0,
  ST_TET, COLOR1, EC, EB, EL, N0,
  ST_TET, COLOR1, EF, EJ, EE, N0,
 // Case #92: (cloned #30)
 // Case #93: (cloned #61)
 // Case #94: (cloned #91)
 // Case #95: Unique case #19
  ST_TET, COLOR0, EK, EH, EG, P7,
  ST_TET, COLOR0, EE, EJ, EF, P5,
  ST_PNT, 0, COLOR1, 6, P0, P1, P2, P3, P4, P6,
  ST_PNT, 1, COLOR1, 6, P4, EE, EF, P6, EG, EH, 
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, P2, P1, EJ, N0,
  ST_TET, COLOR1, P2, EJ, EF, N0,
  ST_TET, COLOR1, P2, EF, P6, N0,
  ST_TET, COLOR1, P3, P0, P1, N0,
  ST_TET, COLOR1, P3, P1, P2, N0,
  ST_TET, COLOR1, N1, EG, P6, N0,
  ST_TET, COLOR1, N1, P6, EF, N0,
  ST_TET, COLOR1, N1, EF, EE, N0,
  ST_TET, COLOR1, N1, EE, P4, N0,
  ST_TET, COLOR1, N1, P4, EH, N0,
  ST_TET, COLOR1, N1, EH, EG, N0,
  ST_TET, COLOR1, P2, P6, EG, N0,
  ST_TET, COLOR1, P2, EG, EK, N0,
  ST_TET, COLOR1, P2, EK, P3, N0,
  ST_TET, COLOR1, P0, P4, EE, N0,
  ST_TET, COLOR1, P0, EE, EJ, N0,
  ST_TET, COLOR1, P0, EJ, P1, N0,
  ST_TET, COLOR1, EH, EK, EG, N0,
  ST_TET, COLOR1, EF, EJ, EE, N0,
 // Case #96: (cloned #3)
 // Case #97: (cloned #21)
 // Case #98: (cloned #7)
 // Case #99: (cloned #23)
 // Case #100: (cloned #7)
 // Case #101: (cloned #30)
 // Case #102: (cloned #15)
 // Case #103: (cloned #31)
 // Case #104: (cloned #21)
 // Case #105: (cloned #60)
 // Case #106: (cloned #30)
 // Case #107: (cloned #61)
 // Case #108: (cloned #23)
 // Case #109: (cloned #61)
 // Case #110: (cloned #31)
 // Case #111: (cloned #63)
 // Case #112: (cloned #7)
 // Case #113: (cloned #23)
 // Case #114: (cloned #27)
 // Case #115: (cloned #31)
 // Case #116: (cloned #23)
 // Case #117: (cloned #61)
 // Case #118: (cloned #31)
 // Case #119: (cloned #63)
 // Case #120: (cloned #30)
 // Case #121: (cloned #61)
 // Case #122: (cloned #91)
 // Case #123: (cloned #95)
 // Case #124: (cloned #61)
 // Case #125: Unique case #20
  ST_TET, COLOR0, EH, EG, EK, P7,
  ST_TET, COLOR0, EA, EB, EJ, P1,
  ST_PNT, 0, COLOR1, 6, P1, P2, P3, P4, P5, P6,
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, P6, P2, EB, N0,
  ST_TET, COLOR1, P6, EB, EJ, N0,
  ST_TET, COLOR1, P6, EJ, P5, N0,
  ST_TET, COLOR1, P3, P0, EA, N0,
  ST_TET, COLOR1, P3, EA, EB, N0,
  ST_TET, COLOR1, P3, EB, P2, N0,
  ST_TET, COLOR1, P5, P4, EH, N0,
  ST_TET, COLOR1, P5, EH, EG, N0,
  ST_TET, COLOR1, P5, EG, P6, N0,
  ST_TET, COLOR1, P2, P6, EG, N0,
  ST_TET, COLOR1, P2, EG, EK, N0,
  ST_TET, COLOR1, P2, EK, P3, N0,
  ST_TET, COLOR1, P4, P5, EJ, N0,
  ST_TET, COLOR1, P4, EJ, EA, N0,
  ST_TET, COLOR1, P4, EA, P0, N0,
  ST_TET, COLOR1, EA, EJ, EB, N0,
  ST_TET, COLOR1, EH, EK, EG, N0,
 // Case #126: (cloned #95)
 // Case #127: Unique case #21
  ST_TET, COLOR0, EH, EG, EK, P7,
  ST_PNT, 0, COLOR1, 7, P0, P1, P2, P3, P4, P5, P6, 
  ST_TET, COLOR1, P0, P3, EK, N0,
  ST_TET, COLOR1, P0, EK, EH, N0,
  ST_TET, COLOR1, P0, EH, P4, N0,
  ST_TET, COLOR1, P2, P1, P5, N0,
  ST_TET, COLOR1, P2, P5, P6, N0,
  ST_TET, COLOR1, P3, P0, P1, N0,
  ST_TET, COLOR1, P3, P1, P2, N0,
  ST_TET, COLOR1, P5, P4, EH, N0,
  ST_TET, COLOR1, P5, EH, EG, N0,
  ST_TET, COLOR1, P5, EG, P6, N0,
  ST_TET, COLOR1, P2, P6, EG, N0,
  ST_TET, COLOR1, P2, EG, EK, N0,
  ST_TET, COLOR1, P2, EK, P3, N0,
  ST_TET, COLOR1, P0, P4, P5, N0,
  ST_TET, COLOR1, P0, P5, P1, N0,
  ST_TET, COLOR1, EH, EK, EG, N0,
 // Case #255: Unique case #22
  ST_PNT, 0, COLOR1, 8, P0, P1, P2, P3, P4, P5, P6, P7,
  ST_TET, COLOR1, P3, P4, P0, N0,
  ST_TET, COLOR1, P3, P7, P4, N0,
  ST_TET, COLOR1, P2, P1, P5, N0,
  ST_TET, COLOR1, P2, P5, P6, N0,
  ST_TET, COLOR1, P3, P0, P1, N0,
  ST_TET, COLOR1, P3, P1, P2, N0,
  ST_TET, COLOR1, P7, P5, P4, N0,
  ST_TET, COLOR1, P7, P6, P5, N0,
  ST_TET, COLOR1, P3, P6, P7, N0,
  ST_TET, COLOR1, P3, P2, P6, N0,
  ST_TET, COLOR1, P0, P4, P5, N0,
  ST_TET, COLOR1, P0, P5, P1, N0,
};
