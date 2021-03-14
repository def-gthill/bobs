
# ==========
# Numbers
# ==========

def preferred(limit=None):
    """
    Generates a sequence of "preferred numbers".
    
    These are nice round numbers that ascend in a
    roughly exponential pattern. The current
    implementation (and the default in all future
    versions) is the sequence 1, 2, 3, 5, 7, 10,
    15, 20, 30, 45, 70, and then repeating the
    numbers 10 to 70 multiplied by each successive
    power of 10.
    
    If a limit is specified, the sequence will stop
    at the last preferred number that doesn't exceed
    that limit. Otherwise, the sequence will be
    infinite.
    """
    for n in [1, 2, 3, 5, 7]:
        if n > limit:
            return
        yield n
    p = 0
    while True:
        for a in [10, 15, 20, 30, 45, 70]:
            n = a * 10 ** p
            if n > limit:
                return
            yield n
        p += 1
            
