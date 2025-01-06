require 'image'
paths.dofile('dataset.lua')

matio=require 'matio'
opt.data = './PATH-TO-THE-TRAINING-DATA/'

if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local ncdataload = opt.ncdataload
local nc = opt.nc
local sampleSize = {ncdataload, opt.fineSizeH, opt.fineSizeW}

local function loadImage(path)
local inp = image.load(path, nc, 'float')
local tgpath='./PATH-TO-GROUND-TRUTH/images/' .. paths.basename(path)
local target = image.load(tgpath, nc, 'float')

if (target:dim() <3) then
	target = target:view(1, target:size(1), target:size(2))
end

if (inp:dim() <3) then
	inp = inp:view(1, inp:size(1), inp:size(2))
end

assert(target:size(1) == 1)
assert(inp:size(1) == 1)
  
local input_src= torch.cat(inp, target, 1)
 
input_src=input_src:float() 
local angle = torch.uniform(-5, 5)--in degrees
angle = (angle/180)*3.1415926535389--in radians
input_src[{{1},{},{}}]=image.rotate(input_src[{{1},{},{}}], angle)
input_src[{{2},{},{}}]=image.rotate(input_src[{{2},{},{}}], angle)
input_ht=input_src:size(2)
input_wt=input_src:size(3)

local input= torch.Tensor(opt.ncdataload, opt.fineSizeH, opt.fineSizeW)
  
local h1 = math.ceil(torch.uniform(1e-2, input_ht-opt.fineSizeH))
local w1 = math.ceil(torch.uniform(1e-2, input_wt-opt.fineSizeW))
input[{{1},{},{}}] = image.crop( input_src[{{1},{},{}}], w1, h1, w1 + opt.fineSizeW, h1 + opt.fineSizeH)
input[{{2},{},{}}] = image.crop(input_src[{{2},{},{}}], w1, h1, w1 + opt.fineSizeW, h1 + opt.fineSizeH)
assert(input:size(3) == opt.fineSizeW)
assert(input:size(2) == opt.fineSizeH) 
   
if math.random() > 0.5 then
	input[{{1},{},{}}] = image.hflip(input[{{1},{},{}}])
	input[{{2},{},{}}] = image.hflip(input[{{2},{},{}}])
end
	 	
input[{{1},{},{}}][torch.eq(input[{{1},{},{}}],0)] = -0.5  
    
scale=torch.uniform(1,1.4)
input[{{1},{},{}}] = input[{{1},{},{}}] / scale
input[{{2},{},{}}] = input[{{2},{},{}}] / scale
input[{{1},{},{}}][torch.le(input[{{1},{},{}}], 0)]=-0.5 
input[{{1},{},{}}]=input[{{1},{},{}}]:float()
input[{{2},{},{}}]=input[{{2},{},{}}]:float()

     return input
end

 
local mean,std
 
local trainHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
    
   local out = input:clone()  
  
   return out
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then   
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.sampleSize = {ncdataload, sampleSize[2], sampleSize[3]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      sampleSize = {ncdataload, sampleSize[2], sampleSize[3]},
      split = 100,  
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()
 
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end

