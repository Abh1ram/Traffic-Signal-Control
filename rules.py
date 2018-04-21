from sets import *

IF, THEN = 0, 1

rule0 = [
    {IF: (time_set[VSHORT], arrival_set[NONE].mt(), queue_set[VSMALL].fany())},
    {IF: (time_set[SHORT], arrival_set[VFEW].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[MEDIUM], arrival_set[VFEW].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[LONG], arrival_set[MEDIUM].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[VLONG], arrival_set[MANY].mt(), queue_set[VSMALL].lt())},
]

rule1 = [
    {IF: (time_set[VSHORT], arrival_set[NONE].mt(), queue_set[VSMALL].fany())},
    {IF: (time_set[SHORT], arrival_set[VFEW].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[MEDIUM], arrival_set[FEW].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[LONG], arrival_set[MEDIUM].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[VLONG], arrival_set[MANY].mt(), queue_set[VSMALL].lt())},
]

rule2 = [
    {IF: (time_set[VSHORT], arrival_set[NONE].mt(), queue_set[VSMALL].fany())},
    {IF: (time_set[SHORT], arrival_set[VFEW].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[MEDIUM], arrival_set[FEW].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[LONG], arrival_set[MEDIUM].mt(), queue_set[VSMALL].lt())},
    {IF: (time_set[VLONG], arrival_set[MANY].mt(), queue_set[SMALL].lt())},
]

rule3 = [
    {IF: (time_set[VSHORT], arrival_set[NONE].mt(), queue_set[VSMALL].fany())},
    {IF: (time_set[SHORT], arrival_set[VFEW].mt(), queue_set[SMALL_PLUS].lt())},
    {IF: (time_set[MEDIUM], arrival_set[MEDIUM].mt(), queue_set[SMALL_PLUS].lt())},
    {IF: (time_set[LONG], arrival_set[MANY].mt(), queue_set[MEDIUM].lt())},
    {IF: (time_set[VLONG], arrival_set[VMANY].mt(), queue_set[LONG].lt())},
]

rule4 = [
    {IF: (time_set[VSHORT], arrival_set[NONE].mt(), queue_set[VSMALL].fany())},
    {IF: (time_set[SHORT], arrival_set[VFEW].mt(), queue_set[LONG].lt())},
    {IF: (time_set[MEDIUM], arrival_set[MEDIUM].mt(), queue_set[LONG].lt())},
    {IF: (time_set[LONG], arrival_set[VMANY].mt(), queue_set[VLONG].lt())},
    {IF: (time_set[VLONG], arrival_set[VMANY].mt(), queue_set[VLONG].lt())},
]

RULES = [
    rule0, 
    rule1,
    rule2,
    rule3,
    rule4,
]