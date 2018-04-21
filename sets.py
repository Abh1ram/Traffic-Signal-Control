from fuzzy import *

# Time Sets
VSHORT, SHORT, PAD, MEDIUM, LONG, VLONG = [i for i in range(6)]
time_set = {
    VSHORT : ListSet([1, 0.5] + [0] * 8, 1),
    SHORT : ListSet([0, 0.5, 1, 0.5] + [0] * 6, 1),
    MEDIUM : ListSet([0] * 3 + [0.5, 1, 0.5] + [0] * 4, 1),
    LONG : ListSet([0] * 5 + [0.5, 1, 0.5] + [0] * 2, 1),
    VLONG : ListSet([0] * 7 + [0.5, 1, 1], 1),
}


# Arrival sets
# a few == VFEW, few == FEW
NONE, VFEW, FEW, MEDIUM, MANY, VMANY = [i for i in range(6)]
arrival_set = {
    NONE : ListSet([0.5, 0.2, 0.1] + [0] * 7, 1),
    VFEW : ListSet([1, 0.5, 0.2, 0.1] + [0] * 6, 1),
    FEW : ListSet([0.5, 1, 0.5, 0.2, 0.1] + [0] * 5, 1),
    MEDIUM : ListSet([0.2, 0.5, 1, 0.5, 0.2, 0.1] + [0] * 4, 1),
    MANY : ListSet([0.1, 0.2, 0.5, 1, 0.5, 0.2, 0.1] + [0] * 3, 1),
    VMANY : ListSet([0, 0.1, 0.2, 0.5, 1, 0.5, 0.2, 0.1] + [0] * 2, 1)
}


# Queue Sets
VSMALL, SMALL, SMALL_PLUS, MEDIUM, LONG, VLONG = [i for i in range(6)]
queue_set = {
    VSMALL : ListSet([0] + [0.5, 0.7, 0.9, 1, 0.9, 0.7, 0.5] + [0] * 21, 4),
    SMALL : ListSet([0] * 5 + [0.5, 0.7, 0.9, 1, 0.9, 0.7, 0.5] + [0] * 17, 4),
    SMALL_PLUS : ListSet([0] * 9 + [0.5, 0.7, 0.9, 1, 0.9, 0.7, 0.5]
            + [0] * 13, 4),
    MEDIUM : ListSet([0] * 13 + [0.5, 0.7, 0.9, 1, 0.9, 0.7, 0.5] + [0] * 9, 4),
    LONG : ListSet([0] * 17 + [0.5, 0.7, 0.9, 1, 0.9, 0.7, 0.5] + [0] * 5, 4),
    VLONG : ListSet([0] * 21 + [0.5, 0.7, 0.9, 1, 0.9, 0.7, 0.5] + [0], 4),
}




if __name__ == "__main__":
    # Tests here
    pass