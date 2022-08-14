function X = generate_ring_input(num)
% generate input for the ring model
    radius = 1;             % ring manifold radius
    t = num;                % tota number of samples
    sep = 2*pi/t;
    X = [radius*cos(0:sep:2*pi-sep);radius*sin(0:sep:2*pi-sep)];
end