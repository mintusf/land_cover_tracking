download_icon_url = "https://leafletjs.com/examples/custom-icons/leaf-green.png"
pred_icon_url = "https://leafletjs.com/examples/custom-icons/leaf-orange.png"
analyze_icon_url = "https://leafletjs.com/examples/custom-icons/leaf-red.png"
shadow_url = "https://leafletjs.com/examples/custom-icons/leaf-shadow.png"
size_3 = [14, 30]
size_2 = [size_3[0] * 1.4, size_3[1] * 1.4]
size_1 = [size_3[0] * 2, size_3[1] * 2]
mult = 0.6


def get_anchor(size):
    return [size_1[0] * mult - 0.5 * size[0] + 15, size_1[1] * mult + size[1] - 35]


download_icon = {
    "iconUrl": download_icon_url,
    "iconSize": size_1,  # size of the icon
    "iconAnchor": get_anchor(
        size_1
    ),  # point of the icon which will correspond to marker's location
    "popupAnchor": [
        -3,
        -76,
    ],  # point from which the popup should open relative to the iconAnchor
}

pred_icon = {
    "iconUrl": pred_icon_url,
    "iconSize": size_2,  # size of the icon
    "iconAnchor": get_anchor(
        size_2
    ),  # point of the icon which will correspond to marker's location
    "popupAnchor": [
        -3,
        -76,
    ],  # point from which the popup should open relative to the iconAnchor
}

analyze_icon = {
    "iconUrl": analyze_icon_url,
    "iconSize": size_3,  # size of the icon
    "iconAnchor": get_anchor(
        size_3
    ),  # point of the icon which will correspond to marker's location
    "popupAnchor": [
        -3,
        -76,
    ],  # point from which the popup should open relative to the iconAnchor
}
