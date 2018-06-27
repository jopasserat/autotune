def merge_two_dicts(dict1, dict2):
    """
    from https://stackoverflow.com/a/26853961/470341
    :param dict1:
    :param dict2:
    :return: merged dictionary
    """
    merged_dict = dict1.copy()   # start with x's keys and values
    merged_dict.update(dict2)    # modifies z with y's keys and values & returns None
    return merged_dict
