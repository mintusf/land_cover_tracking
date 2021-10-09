def get_coord_from_feature(feature):
    bounds = feature["properties"]["_bounds"]
    return f"lat {bounds[0]['lat']:.2f}, long {bounds[0]['lng']:.2f}"