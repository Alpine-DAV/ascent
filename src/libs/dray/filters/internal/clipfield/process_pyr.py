def print_array(name, values):
    print("int "+name+"[256] = {")
    idx = 0
    for row in range(32):
        s = "  "
        for col in range(8):
            s = s + "%d, " % values[idx]
            idx = idx + 1
        if row == 31:
            s = s[:-2]

        s = s + " // cases %d - %d" % (row*8, ((row+1)*8)-1)
        print(s)
    print("};")


def is_pyr(A, B):
    for a in A:
        if a not in B:
            return False
    return True

def split_pyr(toks):
    color = toks[1]
    p = toks[2:]
    # Full faces
    if is_pyr(p, ("P0", "P3", "P7", "P4", "N0")):
        t0 = f"  ST_TET, {color}, P3, P4, P0, N0,\n"
        t1 = f"  ST_TET, {color}, P3, P7, P4, N0,\n"
    elif is_pyr(p, ("P2", "P1", "P5", "P6", "N0")):
        t0 = f"  ST_TET, {color}, P2, P1, P5, N0,\n"
        t1 = f"  ST_TET, {color}, P2, P5, P6, N0,\n"
    elif is_pyr(p, ("P0", "P1", "P2", "P3", "N0")):
        t0 = f"  ST_TET, {color}, P3, P0, P1, N0,\n"
        t1 = f"  ST_TET, {color}, P3, P1, P2, N0,\n"
    elif is_pyr(p, ("P7", "P6", "P5", "P4", "N0")):
        t0 = f"  ST_TET, {color}, P7, P6, P5, N0,\n"
        t1 = f"  ST_TET, {color}, P7, P5, P4, N0,\n"
    elif is_pyr(p, ("P3", "P2", "P6", "P7", "N0")):
        t0 = f"  ST_TET, {color}, P3, P2, P6, N0,\n"
        t1 = f"  ST_TET, {color}, P3, P6, P7, N0,\n"
    elif is_pyr(p, ("P4", "P5", "P1", "P0", "N0")):
        t0 = f"  ST_TET, {color}, P0, P4, P5, N0,\n"
        t1 = f"  ST_TET, {color}, P0, P5, P1, N0,\n"
    # Left half-faces
    elif is_pyr(p, ("P0", "ED", "EH", "P4", "N0")):
        t0 = f"  ST_TET, {color}, ED, EH, P4, N0,\n"
        t1 = f"  ST_TET, {color}, ED, P4, P0, N0,\n"
    elif is_pyr(p, ("ED", "P3", "P7", "EH", "N0")):
        t0 = f"  ST_TET, {color}, P3, P7, EH, N0,\n"
        t1 = f"  ST_TET, {color}, P3, EH, ED, N0,\n"
    elif is_pyr(p, ("P0", "P3", "EK", "EI", "N0")):
        t0 = f"  ST_TET, {color}, P3, EI, P0, N0,\n"
        t1 = f"  ST_TET, {color}, P3, EK, EI, N0,\n"
    elif is_pyr(p, ("EI", "EK", "P7", "P4", "N0")):
        t0 = f"  ST_TET, {color}, EK, P7, P4, N0,\n"
        t1 = f"  ST_TET, {color}, EK, P4, EI, N0,\n"
    # Right half-faces
    elif is_pyr(p, ("P2", "EB", "EF", "P6", "N0")):
        t0 = f"  ST_TET, {color}, P2, EB, EF, N0,\n"
        t1 = f"  ST_TET, {color}, P2, EF, P6, N0,\n"
    elif is_pyr(p, ("EB", "P1", "P5", "EF", "N0")):
        t0 = f"  ST_TET, {color}, EB, P1, P5, N0,\n"
        t1 = f"  ST_TET, {color}, EB, P5, EF, N0,\n"
    elif is_pyr(p, ("P2", "P1", "EJ", "EL", "N0")):
        t0 = f"  ST_TET, {color}, P2, P1, EJ, N0,\n"
        t1 = f"  ST_TET, {color}, P2, EJ, EL, N0,\n"
    elif is_pyr(p, ("EL", "EJ", "P5", "P6", "N0")):
        t0 = f"  ST_TET, {color}, EL, EJ, P5, N0,\n"
        t1 = f"  ST_TET, {color}, EL, P5, P6, N0,\n"
    # Bottom half-faces
    elif is_pyr(p, ("P0", "EA", "EC", "P3", "N0")):
        t0 = f"  ST_TET, {color}, P3, P0, EA, N0,\n"
        t1 = f"  ST_TET, {color}, P3, EA, EC, N0,\n"
    elif is_pyr(p, ("EA", "P1", "P2", "EC", "N0")):
        t0 = f"  ST_TET, {color}, EC, EA, P1, N0,\n"
        t1 = f"  ST_TET, {color}, EC, P1, P2, N0,\n"
    elif is_pyr(p, ("P0", "P1", "EB", "ED", "N0")):
        t0 = f"  ST_TET, {color}, ED, P0, P1, N0,\n"
        t1 = f"  ST_TET, {color}, ED, P1, EB, N0,\n"
    elif is_pyr(p, ("ED", "EB", "P2", "P3", "N0")):
        t0 = f"  ST_TET, {color}, P3, ED, EB, N0,\n"
        t1 = f"  ST_TET, {color}, P3, EB, P2, N0,\n"
    # TOP half-faces
    elif is_pyr(p, ("P7", "EG", "EE", "P4", "N0")):
        t0 = f"  ST_TET, {color}, P7, EG, EE, N0,\n"
        t1 = f"  ST_TET, {color}, P7, EE, P4, N0,\n"
    elif is_pyr(p, ("EG", "P6", "P5", "EE", "N0")):
        t0 = f"  ST_TET, {color}, EG, P6, P5, N0,\n"
        t1 = f"  ST_TET, {color}, EG, P5, EE, N0,\n"
    elif is_pyr(p, ("P7", "P6", "EF", "EH", "N0")):
        t0 = f"  ST_TET, {color}, P7, P6, EF, N0,\n"
        t1 = f"  ST_TET, {color}, P7, EF, EH, N0,\n"
    elif is_pyr(p, ("EH", "EF", "P5", "P4", "N0")):
        t0 = f"  ST_TET, {color}, EH, EF, P5, N0,\n"
        t1 = f"  ST_TET, {color}, EH, P5, P4, N0,\n"
    # BACK half-faces
    elif is_pyr(p, ("P3", "EC", "EG", "P7", "N0")):
        t0 = f"  ST_TET, {color}, P3, EC, EG, N0,\n"
        t1 = f"  ST_TET, {color}, P3, EG, P7, N0,\n"
    elif is_pyr(p, ("EC", "P2", "P6", "EG", "N0")):
        t0 = f"  ST_TET, {color}, EC, P2, P6, N0,\n"
        t1 = f"  ST_TET, {color}, EC, P6, EG, N0,\n"
    elif is_pyr(p, ("P3", "P2", "EL", "EK", "N0")):
        t0 = f"  ST_TET, {color}, P3, P2, EL, N0,\n"
        t1 = f"  ST_TET, {color}, P3, EL, EK, N0,\n"
    elif is_pyr(p, ("EK", "EL", "P6", "P7", "N0")):
        t0 = f"  ST_TET, {color}, EK, EL, P6, N0,\n"
        t1 = f"  ST_TET, {color}, EK, P6, P7, N0,\n"
    # FRONT half-faces
    elif is_pyr(p, ("P1", "EA", "EE", "P5", "N0")):
        t0 = f"  ST_TET, {color}, EA, EE, P5, N0,\n"
        t1 = f"  ST_TET, {color}, EA, P5, P1, N0,\n"
    elif is_pyr(p, ("EA", "P0", "P4", "EE", "N0")):
        t0 = f"  ST_TET, {color}, P0, P4, EE, N0,\n"
        t1 = f"  ST_TET, {color}, P0, EE, EA, N0,\n"
    elif is_pyr(p, ("P1", "P0", "EI", "EJ", "N0")):
        t0 = f"  ST_TET, {color}, P0, EI, EJ, N0,\n"
        t1 = f"  ST_TET, {color}, P0, EJ, P1, N0,\n"
    elif is_pyr(p, ("EJ", "EI", "P4", "P5", "N0")):
        t0 = f"  ST_TET, {color}, EI, P4, P5, N0,\n"
        t1 = f"  ST_TET, {color}, EI, P5, EJ, N0,\n"
    else:
        print("BAD CASE: ", toks)

    s = t0 + t1
#    print(f"s={s}")

    return s


def main():
    lines = open("ClipCasesHex.C", "rt").readlines()
    offsets = [0]*256
    counts = [0]*256
    cases = [""]*256
    labels = [""]*256

    i = 0
    reading = False
    casenum = 0
    offset = 0

    while i < len(lines):
        line = lines[i]
        if not reading:
            if line.find("unsigned char clipShapes") != -1:
                reading = True       
        else:
           if line.find("};") != -1:
               reading = False
           elif line.find("// Case #") != -1:
               pos = line.find("// Case #")
               pos2 = line.find(":")
               casenum = int(line[pos+9:pos2])

               labels[casenum] = line[:-1]

               #print("Case ", casenum)
               offsets[casenum] = offset
           else:
               toks = [x.replace(" ","") for x in line.split(",") if (x.replace(" ","") != "") and (x.replace(" ","") != "\n")]
               #print(toks)
               if toks[0] == "ST_PYR":
                   tetstr = split_pyr(toks)
    
                   cases[casenum] = cases[casenum] + tetstr
                   offset = offset + 2 * 6
                   counts[casenum] = counts[casenum] + 2
               else:
                   # Save off the values that make the shape.
                   cases[casenum] = cases[casenum] + line

                   offset = offset + len(toks)
                   counts[casenum] = counts[casenum] + 1

        i = i + 1
#    return
    print("int numClipCasesHex = 256;")
    print_array("numClipShapesHex", counts)
    print_array("clipShapesHex", offsets)
    print("unsigned char clipShapesHex[] = {")
    for i in range(256):
        #print(f" // Case #{i}: Unique case #{i}")
        print(labels[i])
        print(cases[i][:-1])
    print("};")

main()

