function stack_out = multi_channel_max(stack_in, size_out)

%
size_in = size(stack_in);

for d = 1:3
    if size_out(d) ~= size_in(d)
        project_dim = d;
        break
    end
end

project_step = floor(size_in(project_dim)/ size_out(project_dim));

% initialize & shift 'project_dim' to d3
stack_out = shiftdim(imresize3(stack_in, size_out), d);
stack_in_shift = shiftdim(stack_in, d);

for i = 1:size_out(project_dim)
    if i == size_out(project_dim)
        project_range = (i-1)*project_step+1 : size_in(project_dim);
    else
        project_range = (i-1)*project_step+1 : i*project_step;
    end
    stack_out(:,:,i) = max(stack_in_shift(:,:,project_range),[],3);
    
end

% shiftdim so that the output takes the 1st dim as 'project_dim'
stack_out = shiftdim(stack_out, 2);

end