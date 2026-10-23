"""Part table shared by all scripts. Figure faces -Y; figure's LEFT side is +X."""
PARTS = {
    1: 'base', 2: 'croc_R', 3: 'croc_L', 4: 'pants', 5: 'shirt', 6: 'tshirt', 7: 'neck',
    8: 'arm_R', 9: 'arm_L', 10: 'glove_R', 11: 'glove_L', 12: 'head', 13: 'ear_R', 14: 'ear_L',
    15: 'hair', 16: 'cap', 17: 'mask',
}
ID = {v: k for k, v in PARTS.items()}
# preview colours (painted look)
COLOR = {
    'base': (225, 225, 240), 'croc_R': (95, 160, 230), 'croc_L': (95, 160, 230), 'pants': (130, 185, 240),
    'shirt': (150, 200, 245), 'tshirt': (250, 250, 250), 'neck': (245, 200, 170), 'arm_R': (245, 200, 170),
    'arm_L': (245, 200, 170), 'glove_R': (110, 170, 235), 'glove_L': (110, 170, 235), 'head': (245, 205, 175),
    'ear_R': (240, 190, 160), 'ear_L': (240, 190, 160), 'hair': (245, 245, 250), 'cap': (120, 180, 240), 'mask': (255, 255, 255),
    'sock_R': (250, 250, 250), 'sock_L': (250, 250, 250), 'head_hair': (245, 205, 175),
}
# heights (mm) for a 90 mm figure
Z_BASE, Z_CROC, Z_HEM, Z_GLOVE_TOP, Z_HEAD = 6.8, 11.0, 33.0, 39.5, 55.5
