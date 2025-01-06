-- Required libraries
require 'torch'
require 'nn'
require 'optim'
require 'image'
local io = require 'io'
util = paths.dofile('util.lua') 
local matio=require 'matio'

-- Options for the script
opt = {
    nc = 1,        
    gpu = 1,       -- Use GPU if set to 1
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
 
local nc = opt.nc   

-- Load the pretrained model 
local netG = util.load("./chk/DC_chk_Real.t7", opt.gpu)

if opt.gpu > 0 then  
	require 'cunn'
	if pcall(require, 'cudnn') then  
		require 'cudnn'
		netG = util.cudnn(netG)  
	end
	netG:cuda()   
else
	netG:float()  
end

netG:evaluate()  -- Set model to evaluation mode

-- Path to the directory containing test data
local directory ='/home/cvlab/data/test/NYU/' 

-- Iterate through files in the directory

for file in paths.files(directory) do
	print('Processing test image: ' .. file)

	if file:match('%.mat$') then

		-- Load depth data from .mat file
		local testreal_ctx_realNmask0 = matio.load(directory .. file, 'out2ch')
		testreal_ctx_realNmask0 = testreal_ctx_realNmask0:float()  
		testreal_ctx_realNmask = testreal_ctx_realNmask0:clone()
		
		-- Validate image dimensions
		local imgheight = testreal_ctx_realNmask:size(2)
		local imgwidth = testreal_ctx_realNmask:size(3)
		assert(imgheight == 480, "Image height is not 480")
		assert(imgwidth == 640, "Image width is not 640")

		-- Center crop the input data to required dimensions
		local start_row = math.floor((imgheight - 416) / 2) + 1
		local start_col = math.floor((imgwidth - 544) / 2) + 1
		testreal_ctx_realNmask = testreal_ctx_realNmask[{{}, {start_row, start_row + 415}, {start_col, start_col + 543}}]:clone()

		-- Extract raw depth input
		local testreal_ctx = testreal_ctx_realNmask[{{1}, {}, {}}]:clone()
		local metric_max = testreal_ctx:max()  -- Get the maximum depth value for normalization
		testreal_ctx = testreal_ctx * 1000  -- Scale values
		testreal_ctx = testreal_ctx / testreal_ctx:max()  -- Normalize

		-- Extract and process mask
		local mask = testreal_ctx_realNmask[{{2}, {}, {}}]:clone()
		mask = mask:byte()

		-- Create input tensor and apply mask
		local testinput_ctx = torch.Tensor(opt.nc, imgheight, imgwidth)
		testinput_ctx = testreal_ctx:clone()
		testinput_ctx[{{1}, {}, {}}][mask] = -0.5  -- Apply noisy mask
		testinput_ctx = testinput_ctx:cuda()  

		-- Forward pass through the network
		local testfake = netG:forward(testinput_ctx)
		testfake = testfake:float() 
		testreal_ctx = testreal_ctx:float()
		
		-- Overlay predictions on original depth map
		local pred_center_overlaid = testreal_ctx:clone()
		pred_center_overlaid[{{1}, {}, {}}][mask] = testfake[{{1}, {}, {}}][mask]:float()

		-- Save outputs to .mat file and image format
		local fnname, fnextension = file:match("(.+)%.(.+)$")
		local datamat = {
			predout = testfake,
			depth_max = metric_max
		}
		matio.save(directory .. '../Results/Out_' .. fnname .. '.mat', datamat)

		local svImg = image.toDisplayTensor{input = testfake:float()}; 
		image.save(directory .. '../Results/' .. fnname .. '.png', svImg)

		
		-- Clear intermediate variables to save memory
		testfake = nil
		testreal_ctx_realNmask0 = nil
		testreal_ctx = nil
		testinput_ctx = nil
	end
end

-- Clear model from memory
netG = nil 