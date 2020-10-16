def normalize(list_to_norm, start_range, stop_range):
    length = abs(stop_range - start_range)
    min_num = min(list_to_norm)
    max_num = max(list_to_norm)
    normalized = []
    for num in list_to_norm:
        normalized.append(length * ((num - min_num) / (max_num - min_num)) + start_range)
    return normalized

