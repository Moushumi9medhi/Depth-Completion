-- Required libraries
require 'image'
require 'nn'
util = paths.dofile('util.lua')  -- Utility functions for loading models and other operations
torch.setdefaulttensortype('torch.FloatTensor')  -- Set the default tensor type
local matio = require 'matio'  -- Library for loading/saving .mat files

-- Options for the script
opt = {
    net = './chk/DC_chk_90.t7',    -- Path to the pre-trained model
    gpu = 1,                      -- GPU mode: 0 = CPU, 1 = GPU
    nc = 1,                       -- Number of channels in the input images
}

-- Overwrite options with environment variables if they exist
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- Load the pre-trained model
net = util.load(opt.net, opt.gpu)
net:evaluate() -- Set the network to evaluation mode


-- Directory containing corrupted depth maps (input images)
local input_dir = '/path-to-corrupted depth maps'
-- Directory to save the results
local output_dir = '/Results'
-- Mask file path
local mask_path = 'MASK90.png'

-- Collect all files with '.png' extension in the input directory
files = {}
for file in paths.files(input_dir) do
    if file:find('png' .. '$') then
        table.insert(files, input_dir .. '/' .. file)
    end
end

-- Check if any valid files were found
if #files == 0 then
    error('The given directory does not contain any files of type: png')
end

-- Load all images from the file list
images = {}
for i, file in ipairs(files) do
    -- Load each image  
    table.insert(images, image.load(file, opt.nc, 'float'))
end

-- Load the binary mask  
local mask = image.load(mask_path)
mask = mask:byte() -- Convert the mask to binary (byte tensor)



-- Process each image
for ldimg = 1, #images do  
    local image_ctx = images[ldimg]   
    print(ldimg .. '   :' .. paths.basename(files[ldimg]))   

    local imgheight = image_ctx:size(2)
    local imgwidth = image_ctx:size(3)
 
    local input_image_ctx = torch.Tensor(opt.nc, imgheight, imgwidth)

    -- Move the model and input to GPU if GPU mode is enabled
    if opt.gpu > 0 then  
        require 'cunn'
        if pcall(require, 'cudnn') then  
            require 'cudnn'
            net = util.cudnn(net)  
        end
        net:cuda()
        input_image_ctx = input_image_ctx:cuda()
    else
        net:float()  -- Use CPU
    end

    -- Clone the image as the input
    input_image_ctx = image_ctx:clone()

    -- Apply the mask: set masked pixels to -0.5
    input_image_ctx[{{1}, {}, {}}][mask] = -0.5  

    -- Perform prediction
    local pred_dep
    if opt.noiseGen then
        pred_dep = net:forward({input_image_ctx, noise})
    else
        input_image_ctx = input_image_ctx:cuda()  -- Ensure input is on GPU
        pred_dep = net:forward(input_image_ctx)
    end
 
    pred_dep = pred_dep:resize(pred_dep:size(2), pred_dep:size(3), pred_dep:size(4))
 
    local pred_depth = image_ctx:clone()
    pred_depth[{{1}, {}, {}}][mask] = pred_dep[{{1}, {}, {}}][mask]:float()

    -- Save results
    local basename = paths.basename(files[ldimg])
    matio.save(output_dir .. '/' .. basename:sub(1, #basename - 4) .. '.mat', {
        GT = image_ctx,             -- Ground truth image
        OutputN = pred_dep:float(), -- Predicted depth map (only masked area)
        OutputO = pred_depth:float(), -- Predicted depth map (overlayed on original)
        Maskedimage = input_image_ctx:float() -- Input image with mask applied
    })

    --  Free memory
    pred_dep = nil
    pred_depth = nil
    input_image_ctx = nil
    image_ctx = nil
end

print('Processing completed successfully!')

