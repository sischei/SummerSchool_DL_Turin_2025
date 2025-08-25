function mat2np(m, file, dtype)
%MAT2PKL saves a MATLAB array into a pickled numpy array
%
%  mat2pkl(m, file, dtype) saves a 1-d or 2-d array and into a pickled 
%    numpy array named 'file' in specified dtype. 
%  
%  Currently, only dypte 'int8', 'int16', 'int32' and 'float64' are 
%  supported.
%
% Copyright, Chao-Yuan Yeh, 2016
%

fout = fopen(file, 'wb');
% *consult pickle.py source file for file structure.
% File structure : headbytes + shapebytes + byte_after_shape + dtypebytes +
% byte_after_dtype + dsizebyte + data + tailbyte

headbytes = [128 3 99 110 117 109 112 121 46 99 111 114 101 46 109 117 ...
   108 116 105 97 114 114 97 121 10 95 114 101 99 111 110 115 116 114 117 ...
   99 116 10 113 0 99 110 117 109 112 121 10 110 100 97 114 114 97 121 10 ...
   113 1 75 0 133 113 2 67 1 98 113 3 135 113 4 82 113 5 40];

fwrite(fout, headbytes, 'uint8');

% b75 is the beginning byte of shape records. for shape (1,) this
% is 75, 1, 75, 1. for shape (2,1), this is 75, 1, 75, 2, 75, 1. For
% dimension (2,2), this is 75, 1, 75, 2, 75, 2

fwrite(fout, [75 1], 'uint8');

if size(m, 2) == 1
    if size(m, 1) < 2^8
        fwrite(fout, [75, size(m,1) 133], 'uint8');
    elseif size(m,1) < 2^16
        fwrite(fout, 77, 'uint8');
        fwrite(fout, size(m, 1), 'uint16', 'l');
        fwrite(fout, 134, 'uint8');
    end
else
    if size(m, 1) < 2^8
        fwrite(fout, [75, size(m,1)], 'uint8');
    elseif size(m,1) < 2^16
        fwrite(fout, 77, 'uint8');
        fwrite(fout, size(m, 1), 'uint16', 'l');
    end
    
    if size(m, 2) < 2^8
        fwrite(fout, [75, size(m, 2) 134], 'uint8');
    elseif size(m, 2) < 2^16
        fwrite(fout, 77, 'uint8');
        fwrite(fout, size(m, 2), 'uint16', 'l');
        fwrite(fout, 134, 'uint8');
    end
end

byte_after_shape = [113 6 99 110 117 109 112 121 10 100 116 121 112 101 ...
     10 113 7 88 2 0 0 0];
fwrite(fout, byte_after_shape, 'uint8');

dtype_type = regexpi(dtype, '\D*', 'match');
dtype_type = dtype_type{1};
dtype_size = str2double(regexp(dtype, '\d*', 'match'))/8;

if strfind(dtype_type, 'i')
    if strfind(dtype_type, 'u')
        dtype_type = 'u' + 0; % cast it into integer type
    else
        dtype_type = 'i' + 0;
    end
elseif strfind(dtype_type, 'f')
    dtype_type = 'f' + 0;
end

bytes_after_dtype = [113 8 75 0 75 1 135 113 9 82 113 10 40 75 3 88 1 0 0 ...
    0 60 113 11 78 78 78 74 255 255 255 255 74 255 255 255 255 75 0 116 ...
    113 12 98 137];

fwrite(fout, [dtype_type, num2str(dtype_size) + 0, ...
    bytes_after_dtype], 'uint8');

dsizebytes = size(m, 1) * size(m, 2) * dtype_size ;

if dsizebytes < 2^8
    fwrite(fout, [67 dsizebytes], 'uint8');
else
    fwrite(fout, 66, 'uint8')
    if dsizebytes < 2^16
        fwrite(fout, dsizebytes, 'uint16', 'l');
    elseif dsizebytes < 2^32
        fwrite(fout, dsizebytes, 'uint32', 'l');
    end
end

% data is stored by rows. [[1,2],[3,4]) is stored as 1,2,3,4
% data is stored imeediately before tailbytes 

data = m';

switch dtype
    case {'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'}
        fwrite(fout, data(:), dtype, 'l');
    case 'float64'
        fwrite(fout, data(:), 'float64', 'a');
end

tailbytes = [113 13 116 113 14 98 46];

fwrite(fout, tailbytes, 'uint8');
fclose(fout);

end