import numpy
from functools import reduce

#return a vector pointing in the same direction as vec with length 1
def normalize(vec):
    return vec / numpy.sqrt((vec**2).sum())
    
#rotates p0 by radians around the axis formed by p1 and p2
#source: http://paulbourke.net/geometry/rotate/
def rotate(p0, p1, p2, radians):
    #unit normalized vector formed by p2 - p1
    normalized_position = normalize(p2 - p1)
    a = normalized_position[0]
    b = normalized_position[1]
    c = normalized_position[2]

    #length of projection of normalized_position on the xy plane
    d = numpy.sqrt(b**2 + c**2)
    
    #translate rotation axis to origin
    translate = numpy.array([
        [1, 0, 0, -p1[0]],
        [0, 1, 0, -p1[1]],
        [0, 0, 1, -p1[2]],
        [0, 0, 0,      1]
    ])
    translate_inv = numpy.array([
        [1, 0, 0, p1[0]],
        [0, 1, 0, p1[1]],
        [0, 0, 1, p1[2]],
        [0, 0, 0,     1]
    ])
    #rotate about x axis so the rotation axis lies in the xz plane
    rot_x = numpy.array([
        [1,   0,    0, 0],
        [0, c/d, -b/d, 0],
        [0, b/d,  c/d, 0],
        [0,   0,    0, 1]
    ])
    rot_x_inv = numpy.array([
        [1,    0,   0, 0],
        [0,  c/d, b/d, 0],
        [0, -b/d, c/d, 0],
        [0,    0,   0, 1]
    ])
    #rotate about y axis so the roation axis lies on the z axis
    rot_y = numpy.array([
        [d, 0, -a, 0],
        [0, 1,  0, 0],
        [a, 0,  d, 0],
        [0, 0,  0, 1]
    ])
    rot_y_inv = numpy.array([
        [ d, 0, a, 0],
        [ 0, 1, 0, 0],
        [-a, 0, d, 0],
        [ 0, 0, 0, 1]
    ])
    #rotate about z axis
    rot_z = numpy.array([
        [numpy.cos(radians), -numpy.sin(radians), 0, 0],
        [numpy.sin(radians), numpy.cos(radians), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    #multiply by all the forward transforms followed by reverse transforms
    #return translate_inv @ rot_x_inv @ rot_y_inv @ rot_z @ rot_y @ rot_x @ translate @ p0
    return reduce(numpy.dot, [translate_inv, rot_x_inv, rot_y_inv, rot_z, rot_y, rot_x, translate, p0])



def look_at(info, position, look_at, up):
    render_info = info['images'][0]

    #TODO figure out why it needs a numpy array
    render_info['camera']['position'] = numpy.array(position)
    render_info['camera']['look_at'] = numpy.array(look_at)
    render_info['camera']['up'] = numpy.array(up)

    render_info['camera']['zoom'] = 0

    return render_info


def move_up(info, multiplier=.1):
    render_info = info['images'][0]

    position = render_info['camera']['position']
    look_at = render_info['camera']['look_at']
    up = render_info['camera']['up']
    scene_magnitude = numpy.sqrt(
                        (render_info['scene_bounds'][3] - render_info['scene_bounds'][0])**2
                        + (render_info['scene_bounds'][4] - render_info['scene_bounds'][1])**2
                        + (render_info['scene_bounds'][5] - render_info['scene_bounds'][2])**2
                    )

    render_info['camera']['position'] = numpy.add(position, numpy.array(up) * multiplier * scene_magnitude)
    render_info['camera']['look_at'] = numpy.add(look_at, numpy.array(up) * multiplier * scene_magnitude)

    render_info['camera']['zoom'] = 0

    return render_info
    
def move_forward(info, multiplier=.1):
    render_info = info['images'][0]

    position = render_info['camera']['position']
    look_at = render_info['camera']['look_at']
    up = render_info['camera']['up']
    scene_magnitude = numpy.sqrt(
                        (render_info['scene_bounds'][3] - render_info['scene_bounds'][0])**2
                        + (render_info['scene_bounds'][4] - render_info['scene_bounds'][1])**2
                        + (render_info['scene_bounds'][5] - render_info['scene_bounds'][2])**2
                    )
                    
    forward = normalize(numpy.subtract(look_at, position))

    render_info['camera']['position'] = numpy.add(position, forward * multiplier * scene_magnitude)
    render_info['camera']['look_at'] = numpy.add(look_at, forward * multiplier * scene_magnitude)

    render_info['camera']['zoom'] = 0

    return render_info

def move_right(info, multiplier=.1):
    render_info = info['images'][0]

    position = render_info['camera']['position']
    look_at = render_info['camera']['look_at']
    up = render_info['camera']['up']
    scene_magnitude = numpy.sqrt(
                        (render_info['scene_bounds'][3] - render_info['scene_bounds'][0])**2
                        + (render_info['scene_bounds'][4] - render_info['scene_bounds'][1])**2
                        + (render_info['scene_bounds'][5] - render_info['scene_bounds'][2])**2
                    )
                    
    forward = normalize(numpy.subtract(look_at, position))
    right = normalize(numpy.cross(forward, up))

    render_info['camera']['position'] = numpy.add(position, right * multiplier * scene_magnitude)
    render_info['camera']['look_at'] = numpy.add(look_at, right * multiplier * scene_magnitude)

    render_info['camera']['zoom'] = 0

    return render_info

#rotate the camera about the direction we are looking in 
def roll(info, radians=2*numpy.pi/36):
    render_info = info['images'][0]

    position = render_info['camera']['position']
    look_at = render_info['camera']['look_at']
    up = render_info['camera']['up']

    #point for up relative to position 4-vector ready for transformation
    pup = numpy.append(up + position, 1)

    #rotated up vector
    res_up = rotate(pup, position, look_at, radians)[:-1] - position

    render_info['camera']['up'] = res_up

    render_info['camera']['zoom'] = 0
    
    return render_info

def pitch(info, radians=2*numpy.pi/36):
    render_info = info['images'][0]

    position = render_info['camera']['position']
    look_at = render_info['camera']['look_at']
    up = render_info['camera']['up']

    forward = normalize(numpy.subtract(look_at, position))
    right = normalize(numpy.cross(forward, up))

    #look-at 4-vector ready for transformation
    plook_at = numpy.append(look_at, 1)

    #rotated look_at point
    res_look_at = rotate(plook_at, position, position + right, radians)[:-1]
    res_up = normalize(numpy.cross(right, res_look_at - position))

    render_info['camera']['look_at'] = res_look_at
    render_info['camera']['up'] = res_up

    render_info['camera']['zoom'] = 0
    
    return render_info


def yaw(info, radians=2*numpy.pi/36):
    render_info = info['images'][0]

    position = render_info['camera']['position']
    look_at = render_info['camera']['look_at']
    up = render_info['camera']['up']

    #look-at 4-vector ready for transformation
    plook_at = numpy.append(look_at, 1)

    #rotated up vector
    res_look_at = rotate(plook_at, position, position + up, radians)[:-1]

    render_info['camera']['look_at'] = res_look_at

    render_info['camera']['zoom'] = 0

    return render_info

utils_dict = {
    'normalize': normalize,
    'rotate': rotate,
    'look_at': look_at,
    'move_up': move_up,
    'move_forward': move_forward,
    'move_right': move_right,
    'roll': roll,
    'pitch': pitch,
    'yaw': yaw,
}
