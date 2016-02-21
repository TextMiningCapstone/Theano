def build_wv():
    words = {}
    vectors = []

    with open("./vocab.txt") as f:
        i = 0
        for line in f:
            words[line.strip()] = i
            i += 1

    with open("./wordVectors.txt") as f:
        for line in f:
            nums = line.strip().split(" ")
            vectors.append([float(i) for i in nums])

    return words, vectors

def get_data(words):
    data = []
    label = []
    label_hotv = []
    with open("./train") as f:
        for line in f:
            if line.strip() == "":
                continue
            elems = line.strip().split("\t")
            if len(elems) != 2:
                continue
            if not elems[0] in words:
                continue
            data.append(str(elems[0]))
            label.append(elems[1])

            hv = [0.0, 0.0, 0.0, 0.0, 0.0]

            if elems[1] == "O":
                hv[0] = 1.0
            elif elems[1] == "PER":
                hv[1] = 1.0
            elif elems[1] == "LOC":
                hv[2] = 1.0
            elif elems[1] == "MISC":
                hv[3] = 1.0
            elif elems[1] == "ORG":
                hv[4] = 1.0

            label_hotv.append(hv)

    return data, label, label_hotv

