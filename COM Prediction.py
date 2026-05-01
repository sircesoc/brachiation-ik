def COM_Prediction(theta_1_degrees, theta_2_degrees, theta_11_degrees, theta_12_degrees, theta_tail_degrees):
    # Inputs should be in degrees #
    import math
    # Right Arm Values #
    #theta_1_degrees = 15 # in degrees
    #theta_2_degrees = 0 # in degrees
    #theta_tail_degrees = 10 # in degrees
    theta_2_absolute_degrees = theta_1_degrees + theta_2_degrees
    theta_1 = theta_1_degrees * (math.pi / 180) # in radians
    theta_2 = theta_2_degrees * (math.pi / 180) # in radians
    theta_2_absolute = theta_1 + theta_2 # in radians

    # Left Arm Values #
    #theta_11_degrees = 30 # in degrees
    #theta_12_degrees = 20 # in degrees
    theta_12_absolute_degrees = theta_11_degrees + theta_12_degrees
    theta_11 = theta_11_degrees * (math.pi / 180) # in radians
    theta_12 = theta_12_degrees * (math.pi / 180) # in radians
    theta_12_absolute = theta_11 + theta_12 # in radians

    theta_tail = theta_tail_degrees * (math.pi / 180) # in radians

    Link1_length = 3.4 # in inches
    Link2_length = 11.8 # in inches
    body_pivot1_offset = 5.99606299

    # Body #
    m_body = 9.68960433 # in pounds
    x_com_body = 0 # in inches
    y_com_body = -2.22013148 # in inches

    # Tail #
    m_tail = 3.9471154 # in pounds
    tail_mass_offset = 15.3379 # in inches
    tail_angle_stop_degrees = 8.73 # in degrees
    tail_angle_stop = tail_angle_stop_degrees * (math.pi / 180) # in radians
    x_com_tail =  tail_mass_offset * math.cos(theta_tail + tail_angle_stop) # in inches
    y_com_tail = -10 - tail_mass_offset * math.sin(theta_tail + tail_angle_stop) # in inches

    ## Right Arm ##
    
    # Right Arm - Link 1 #
    m_link1 = 2.78627841 # in pounds
    x_com_link1 = -body_pivot1_offset-Link1_length*math.cos(theta_1) # in inches
    y_com_link1 = Link1_length*math.sin(theta_1) # in inches

    # Right Arm - Link 2 #
    m_link2 = 1.16620275 # in pounds
    x_offset_link2_cubed_coefficient = 0.0
    x_offset_link2_squared_coefficient = -0.0001699867194
    x_offset_link2_linear_coefficient = 0.03098551081
    x_offset_link2_constant_coefficient = -0.1653118738

    y_offset_link2_cubed_coefficient = 0.0000009035335573
    y_offset_link2_squared_coefficient = -0.0002522954645
    y_offset_link2_linear_coefficient = 0.00184756179
    y_offset_link2_constant_coefficient = 2.727258043

    x_offset_Link2 = (
        x_offset_link2_cubed_coefficient * (theta_2_absolute_degrees ** 3)
        + x_offset_link2_squared_coefficient * (theta_2_absolute_degrees ** 2)
        + x_offset_link2_linear_coefficient * theta_2_absolute_degrees
        + x_offset_link2_constant_coefficient
    ) # ANGLES NEED TO BE IN RADIANS
    y_offset_Link2 = (
        y_offset_link2_cubed_coefficient * (theta_2_absolute_degrees ** 3)
        + y_offset_link2_squared_coefficient * (theta_2_absolute_degrees ** 2)
        + y_offset_link2_linear_coefficient * theta_2_absolute_degrees
        + y_offset_link2_constant_coefficient
    )

    x_com_link2 = -body_pivot1_offset-Link1_length*math.cos(theta_1) - ((Link2_length/2 + y_offset_Link2)*math.cos(theta_2_absolute)) - x_offset_Link2*math.cos(math.pi / 2 - theta_2_absolute) # in inches
    y_com_link2 = Link1_length*math.sin(theta_1) + (Link2_length/2 + y_offset_Link2)*math.sin(theta_2_absolute) - x_offset_Link2*math.sin(math.pi / 2 - theta_2_absolute) # in inches

    # Right Arm - Gripper #
    m_gripper = 4.08915 # in pounds
    x_gripper_offset = 0.37758314 # in inches
    y_gripper_offset = 3.87674 # in inches
    x_com_gripper = -body_pivot1_offset-Link1_length*math.cos(theta_1) - Link2_length*math.cos(theta_2_absolute) - x_gripper_offset # in inches
    y_com_gripper = Link1_length*math.sin(theta_1) + Link2_length*math.sin(theta_2_absolute) + y_gripper_offset # in inches


    ## Left Arm ##

    # Left Arm - Link 1 #
    m_link1_left = m_link1 # in pounds
    x_com_link1_left = body_pivot1_offset + Link1_length*math.cos(theta_11) # in inches
    y_com_link1_left = Link1_length*math.sin(theta_11) # in inches

    # Left Arm - Link 2 #
    m_link2_left = m_link2 # in pounds
    x_offset_link2_left = (
        x_offset_link2_cubed_coefficient * (theta_12_absolute_degrees ** 3)
        + x_offset_link2_squared_coefficient * (theta_12_absolute_degrees ** 2)
        + x_offset_link2_linear_coefficient * theta_12_absolute_degrees
        + x_offset_link2_constant_coefficient
    ) # in inches
    y_offset_link2_left = (
        y_offset_link2_cubed_coefficient * (theta_12_absolute_degrees ** 3)
        + y_offset_link2_squared_coefficient * (theta_12_absolute_degrees ** 2)
        + y_offset_link2_linear_coefficient * theta_12_absolute_degrees
        + y_offset_link2_constant_coefficient
    ) # in inches
    x_com_link2_left = body_pivot1_offset + Link1_length*math.cos(theta_11) + ((Link2_length/2 + y_offset_link2_left)*math.cos(theta_12_absolute)) + x_offset_link2_left*math.cos(math.pi / 2 - theta_12_absolute) # in inches
    y_com_link2_left = Link1_length*math.sin(theta_11) + (Link2_length/2 + y_offset_link2_left)*math.sin(theta_12_absolute) - x_offset_link2_left*math.sin(math.pi / 2 - theta_12_absolute) # in inches

    # Left Arm - Gripper #
    m_gripper_left = m_gripper # in pounds
    x_gripper_offset_left = x_gripper_offset # in inches
    y_gripper_offset_left = y_gripper_offset # in inches
    x_com_gripper_left = body_pivot1_offset + Link1_length*math.cos(theta_11) + Link2_length*math.cos(theta_12_absolute) + x_gripper_offset_left # in inches
    y_com_gripper_left = Link1_length*math.sin(theta_11) + Link2_length*math.sin(theta_12_absolute) + y_gripper_offset_left # in inches


    # Combined center of mass (body + tail + both arms), same frame as segment COMs (inches, pounds as mass weights)
    m_total = (
        m_body
        + m_tail
        + m_link1
        + m_link2
        + m_gripper
        + m_link1_left
        + m_link2_left
        + m_gripper_left
    )

    x_moment = (
        m_body * x_com_body
        + m_tail * x_com_tail
        + m_link1 * x_com_link1
        + m_link2 * x_com_link2
        + m_gripper * x_com_gripper
        + m_link1_left * x_com_link1_left
        + m_link2_left * x_com_link2_left
        + m_gripper_left * x_com_gripper_left
    )
    y_moment = (
        m_body * y_com_body
        + m_tail * y_com_tail
        + m_link1 * y_com_link1
        + m_link2 * y_com_link2
        + m_gripper * y_com_gripper
        + m_link1_left * y_com_link1_left
        + m_link2_left * y_com_link2_left
        + m_gripper_left * y_com_gripper_left
    )

    x_com_total = x_moment / m_total # in inches
    y_com_total = y_moment / m_total # in inches

    return x_com_total, y_com_total

#print(COM_Prediction(43.10249, 82.79655965, 48.483323, -0.05386829, 162.54))