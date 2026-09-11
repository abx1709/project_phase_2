def print_tqdm(*args, sep=" ", end="\n"):
    """
    A wrapper around tqdm.write to print messages without interfering with the progress bar.
    """
    from tqdm import tqdm
    string = sep.join(str(arg) for arg in args)
    tqdm.write(string, end=end)
    