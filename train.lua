-- require('mobdebug').start() 
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'cunn'
local io = require 'io'
local matio=require 'matio'
util = paths.dofile('util.lua') 


opt = {
   batchSize = 16,        
   loadSize = 256,        
   fineSize = 256,         
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 32,               -- #  of discrim filters in first conv layer
   nc = 1,                 -- # of channels in input -- initially 3
   wtl2 = 0.999,               
   ncdataload = 2,
   nThreads = 1,           -- #  of data loading threads to use 
   niter = 3000,             -- #  of iter at starting learning rate 
    
   beta1 = 0.5,            -- momentum term of adam
   momentum = 0.9,
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on  

   manualSeed = 0,         -- 0 means random seed
   continue_train=1,          -- if continue training, load the latest model: 1: true, 0: false
   save_gen_samples = 0,   -- save generated samples while training. 0 = false
   
 
   
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)



-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(10001, 20000)
end
print("Seed: " .. opt.manualSeed)


torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size()) 

---------------------------------------------------------------------------
-- Initialize network variables
---------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
  
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = opt.nc    
local nz = opt.nz
local nBottleneck = opt.nBottleneck
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 0.9 #Label smoothing
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netD
local netG
if opt.continue_train == 1 then
	print('loading previously trained netG...')
	netG = util.load(string.format("netG_%d.t7", 7), opt.gpu)
	print('loading previously trained netD...')
	netD = util.load(string.format("netD_%d.t7", 7), opt.gpu)
else

	function branch(insert)
		local block = nn.Sequential()
		block:add(insert)
		local parallel = nn.ConcatTable(2)
		parallel:add(nn.Identity())
		parallel:add(block)
		local model = nn.Sequential()
		model:add(parallel)
		model:add(nn.JoinTable(2))
		return model
end


block0= nn.Sequential()
 
block0:add(nn.SpatialDilatedConvolution(ngf * 2, ngf * 4, 3, 3, 1, 1, 2, 2, 2, 2))
block0:add(SpatialBatchNormalization(ngf * 4)):add(nn.ELU(0.8, true))

block0:add(nn.SpatialDilatedConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 4, 4, 4, 4))
block0:add(SpatialBatchNormalization(ngf * 4)):add(nn.ELU(0.8, true))

block0:add(nn.SpatialDilatedConvolution(ngf * 4, ngf * 2, 3, 3, 1, 1, 2, 2, 2, 2))
block0:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
 
block1 = nn.Sequential()
block1_1=nn.Sequential()
block1_1:add(SpatialConvolution(ngf, ngf * 2 , 4, 4, 2, 2, 1, 1))
block1_1:add(SpatialBatchNormalization(ngf * 2)):add(nn.ELU(0.8, true))

block1_2=nn.Sequential()
block1_2:add(SpatialConvolution(ngf * 2 * 2 , ngf * 2 , 3, 3, 1, 1, 1, 1))
block1_2:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU( true))
block1_2:add(SpatialFullConvolution(ngf * 2, ngf , 4, 4, 2, 2, 1, 1))
block1_2:add(SpatialBatchNormalization(ngf )):add(nn.ReLU(true))

block1:add(block1_1) 
block1:add(branch(block0))
block1:add(block1_2) 

netG = nn.Sequential()
netG:add(SpatialConvolution(nc, ngf, 3, 3, 1, 1, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ELU(0.8, true))
netG:add(branch(block1))
netG:add(SpatialConvolution(ngf * 2  , ngf , 3, 3, 1, 1, 1, 1))
netG:add(SpatialBatchNormalization(ngf )):add(nn.ReLU( true))
netG:add(SpatialFullConvolution(ngf, opt.nc, 3, 3, 1, 1, 1, 1))
netG:add(nn.Tanh())

netG:apply(weights_init) 

---------------------------------------------------------------------------
-- Adversarial discriminator net
---------------------------------------------------------------------------
netD = nn.Sequential()  
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
 
netD:add(SpatialConvolution(ndf, ndf *2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
 
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
 
netD:add(SpatialConvolution(ndf * 4, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
 
netD:add(nn.Reshape(16*16*ndf * 4)) 
netD:add(nn.Linear(16*16*ndf * 4,1))
netD:add(nn.Sigmoid())
 
netD:apply(weights_init) 

end
---------------------------------------------------------------------------
-- Loss Metrics
---------------------------------------------------------------------------
local criterion = nn.BCECriterion()

---------------------------------------------------------------------------
-- Setup Solver
---------------------------------------------------------------------------
optimStateG = {
  learningRate = 0.00025,
    weightDecay=0.0001,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = 0.0001, 
   weightDecay=0.001,
   momentum = opt.momentum,
}

---------------------------------------------------------------------------
-- Initialize data variables
---------------------------------------------------------------------------
 
local  mask
local input_ctx = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)--all 0
local input_center = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)--initialized
local input_real_center
if opt.wtl2~=0 then  
    input_real_center = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
end
 
local label = torch.Tensor(opt.batchSize)
local errD, errD_real, errD_fake, errD_realTotal, errD_fakeTotal, MSssim_error, errG_MSssim_EpochAvg, errG_huber, errG_huber_EpochAvg, errG, errG_adv_EpochAvg,  errG_total, errG_total_EpochAvg
local counterD, counterG
  
if pcall(require, 'cudnn') and pcall(require, 'cunn') and opt.gpu>0 then 
end
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
    input_ctx = input_ctx:cuda();  
    label = label:cuda() 
 
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda();      
   if opt.wtl2~=0 then  
   input_real_center = input_real_center:cuda();
   end
end
 
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()


-----------------------------------------MSssim--------------------------
grad = require 'autograd'
n=5  
level=5
mu_1 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_2 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_3 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_4 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_5 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_6 = grad.nn.SpatialAveragePooling(2,2,2,2)
mu_7 = grad.nn.SpatialAveragePooling(2,2,2,2) 

mu_1_2 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_2_2 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_3_2 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_4_2 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_5_2 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_6_2 = grad.nn.SpatialAveragePooling(2,2,2,2)
mu_7_2 = grad.nn.SpatialAveragePooling(2,2,2,2) 

mu_1_3 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_2_3 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_3_3 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_4_3 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_5_3 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_6_3 = grad.nn.SpatialAveragePooling(2,2,2,2)
mu_7_3 = grad.nn.SpatialAveragePooling(2,2,2,2) 

mu_1_4 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_2_4 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_3_4 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_4_4 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_5_4 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_6_4 = grad.nn.SpatialAveragePooling(2,2,2,2)
mu_7_4 = grad.nn.SpatialAveragePooling(2,2,2,2) 

mu_1_5 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_2_5 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_3_5 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_4_5 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_5_5 = grad.nn.SpatialAveragePooling(n,n,1,1) 
mu_6_5 = grad.nn.SpatialAveragePooling(2,2,2,2)
mu_7_5 = grad.nn.SpatialAveragePooling(2,2,2,2) 

cm_1 =  grad.nn.CMulTable()
cm_2 =  grad.nn.CMulTable()
cm_3 =  grad.nn.CMulTable()
cm_4 =  grad.nn.CMulTable()
cm_5 =  grad.nn.CMulTable()
cm_6 =  grad.nn.CMulTable()
cm_7 =  grad.nn.CMulTable()
cm_8 =  grad.nn.CMulTable()

cm_1_2 =  grad.nn.CMulTable()
cm_2_2 =  grad.nn.CMulTable()
cm_3_2 =  grad.nn.CMulTable()
cm_4_2 =  grad.nn.CMulTable()
cm_5_2 =  grad.nn.CMulTable()
cm_6_2 =  grad.nn.CMulTable()
cm_7_2 =  grad.nn.CMulTable()
cm_8_2 =  grad.nn.CMulTable()

cm_1_3 =  grad.nn.CMulTable()
cm_2_3 =  grad.nn.CMulTable()
cm_3_3 =  grad.nn.CMulTable()
cm_4_3 =  grad.nn.CMulTable()
cm_5_3 =  grad.nn.CMulTable()
cm_6_3 =  grad.nn.CMulTable()
cm_7_3 =  grad.nn.CMulTable()
cm_8_3 =  grad.nn.CMulTable()

cm_1_4 =  grad.nn.CMulTable()
cm_2_4 =  grad.nn.CMulTable()
cm_3_4 =  grad.nn.CMulTable()
cm_4_4 =  grad.nn.CMulTable()
cm_5_4 =  grad.nn.CMulTable()
cm_6_4 =  grad.nn.CMulTable()
cm_7_4 =  grad.nn.CMulTable()
cm_8_4 =  grad.nn.CMulTable()

cm_1_5 =  grad.nn.CMulTable()
cm_2_5 =  grad.nn.CMulTable()
cm_3_5 =  grad.nn.CMulTable()
cm_4_5 =  grad.nn.CMulTable()
cm_5_5 =  grad.nn.CMulTable()
cm_6_5 =  grad.nn.CMulTable()
cm_7_5 =  grad.nn.CMulTable()
cm_8_5 =  grad.nn.CMulTable()


cd_1 =  grad.nn.CDivTable()
cd_2 =  grad.nn.CDivTable()
cd_1_2 =  grad.nn.CDivTable()
cd_2_2 =  grad.nn.CDivTable()
cd_1_3 =  grad.nn.CDivTable()
cd_2_3 =  grad.nn.CDivTable()
cd_1_4 =  grad.nn.CDivTable()
cd_2_4 =  grad.nn.CDivTable()
cd_1_5 =  grad.nn.CDivTable()
cd_2_5 =  grad.nn.CDivTable()

MSssim = function (y_im,x_im)
im1 = y_im;
im2 = x_im;

--level 1
       mu_x_1 = mu_1(im2)
       mu_y_1 = mu_2(im1)
       mu_x_sq_1 = cm_1({mu_x_1,mu_x_1})
       mu_y_sq_1 = cm_2({mu_y_1,mu_y_1})
       mu_xy_1 = cm_3({mu_x_1,mu_y_1})
       X_2_1 = cm_4({im2, im2})
       Y_2_1 = cm_5({im1, im1})
       XY_1 = cm_6({im2,im1})
       sigma_x_sq_1 = mu_3(X_2_1)-mu_x_sq_1
       sigma_y_sq_1 = mu_4(Y_2_1)-mu_y_sq_1
       sigma_xy_1 = mu_5(XY_1)-mu_xy_1
       A1_1 = mu_xy_1*2+ 0.0001 
       A2_1 = sigma_xy_1*2+ 0.0009  
       B1_1  = mu_x_sq_1+mu_y_sq_1+ 0.0001 
       B2_1 = sigma_x_sq_1+sigma_y_sq_1+ 0.0009 
       A_1 = cm_7({A1_1,A2_1})
       B_1 = cm_8({B1_1,B2_1})
       ssim_array_1 = torch.mean(cd_1({A_1,B_1}))
       mcs_array_1 = torch.mean(cd_2({A2_1,B2_1}))
       filtered_im1_1 = mu_6(im1)
	   filtered_im2_1 = mu_7(im2)
	   im1_2 = filtered_im1_1 
	   im2_2 = filtered_im2_1
--level 2

       mu_x_2 = mu_1_2(im2_2)
       mu_y_2 = mu_2_2(im1_2)
       mu_x_sq_2 = cm_1_2({mu_x_2,mu_x_2})
       mu_y_sq_2 = cm_2_2({mu_y_2,mu_y_2})
       mu_xy_2 = cm_3_2({mu_x_2,mu_y_2})
       X_2_2 = cm_4_2({im2_2, im2_2})
       Y_2_2 = cm_5_2({im1_2, im1_2})
       XY_2 = cm_6_2({im2_2,im1_2})
       sigma_x_sq_2 = mu_3_2(X_2_2)-mu_x_sq_2
       sigma_y_sq_2 = mu_4_2(Y_2_2)-mu_y_sq_2
       sigma_xy_2 = mu_5_2(XY_2)-mu_xy_2
       A1_2 = mu_xy_2*2+ 0.0001 
       A2_2 = sigma_xy_2*2+ 0.0009  
       B1_2  = mu_x_sq_2+mu_y_sq_2+ 0.0001 
       B2_2 = sigma_x_sq_2+sigma_y_sq_2+ 0.0009 
       A_2 = cm_7_2({A1_2,A2_2})
       B_2 = cm_8_2({B1_2,B2_2})
       ssim_array_2 = torch.mean(cd_1_2({A_2,B_2}))
        mcs_array_2 = torch.mean(cd_2_2({A2_2,B2_2}))
       filtered_im1_2 = mu_6_2(im1_2)
	   filtered_im2_2 = mu_7_2(im2_2)
	   im1_3 = filtered_im1_2 
	   im2_3 = filtered_im2_2

--level 3

       mu_x_3 = mu_1_3(im2_3)
       mu_y_3 = mu_2_3(im1_3)
       mu_x_sq_3 = cm_1_3({mu_x_3,mu_x_3})
       mu_y_sq_3 = cm_2_3({mu_y_3,mu_y_3})
       mu_xy_3 = cm_3_3({mu_x_3,mu_y_3})
       X_2_3 = cm_4_3({im2_3, im2_3})
       Y_2_3 = cm_5_3({im1_3, im1_3})
       XY_3 = cm_6_3({im2_3,im1_3})
       sigma_x_sq_3 = mu_3_3(X_2_3)-mu_x_sq_3
       sigma_y_sq_3 = mu_4_3(Y_2_3)-mu_y_sq_3
       sigma_xy_3 = mu_5_3(XY_3)-mu_xy_3
       A1_3 = mu_xy_3*2+ 0.0001 
       A2_3 = sigma_xy_3*2+ 0.0009  
       B1_3  = mu_x_sq_3+mu_y_sq_3+ 0.0001 
       B2_3 = sigma_x_sq_3+sigma_y_sq_3+ 0.0009 
       A_3 = cm_7_3({A1_3,A2_3})
       B_3 = cm_8_3({B1_3,B2_3})
       ssim_array_3 = torch.mean(cd_1_3({A_3,B_3}))
       mcs_array_3 = torch.mean(cd_2_3({A2_3,B2_3}))
       filtered_im1_3 = mu_6_3(im1_3)
	   filtered_im2_3 = mu_7_3(im2_3)
	   im1_4 = filtered_im1_3 
	   im2_4 = filtered_im2_3

--level 4

       mu_x_4 = mu_1_4(im2_4)
       mu_y_4 = mu_2_4(im1_4)
       mu_x_sq_4 = cm_1_4({mu_x_4,mu_x_4})
       mu_y_sq_4 = cm_2_4({mu_y_4,mu_y_4})
       mu_xy_4 = cm_3_4({mu_x_4,mu_y_4})
       X_2_4 = cm_4_4({im2_4, im2_4})
       Y_2_4 = cm_5_4({im1_4, im1_4})
       XY_4 = cm_6_4({im2_4,im1_4})
       sigma_x_sq_4 = mu_3_4(X_2_4)-mu_x_sq_4
       sigma_y_sq_4 = mu_4_4(Y_2_4)-mu_y_sq_4
       sigma_xy_4 = mu_5_4(XY_4)-mu_xy_4
       A1_4 = mu_xy_4*2+ 0.0001 
       A2_4 = sigma_xy_4*2+ 0.0009  
       B1_4  = mu_x_sq_4+mu_y_sq_4+ 0.0001 
       B2_4 = sigma_x_sq_4+sigma_y_sq_4+ 0.0009 
       A_4 = cm_7_4({A1_4,A2_4})
       B_4 = cm_8_4({B1_4,B2_4})
       ssim_array_4 = torch.mean(cd_1_4({A_4,B_4}))
        mcs_array_4 = torch.mean(cd_2_4({A2_4,B2_4}))
       filtered_im1_4 = mu_6_4(im1_4)
	   filtered_im2_4 = mu_7_4(im2_4)
	   im1_5 = filtered_im1_4 
	   im2_5 = filtered_im2_4

--level 5

       mu_x_5 = mu_1_5(im2_5)
       mu_y_5 = mu_2_5(im1_5)
       mu_x_sq_5 = cm_1_5({mu_x_5,mu_x_5})
       mu_y_sq_5 = cm_2_5({mu_y_5,mu_y_5})
       mu_xy_5 = cm_3_5({mu_x_5,mu_y_5})
       X_2_5 = cm_4_5({im2_5, im2_5})
       Y_2_5 = cm_5_5({im1_5, im1_5})
       XY_5 = cm_6_5({im2_5,im1_5})
       sigma_x_sq_5 = mu_3_5(X_2_5)-mu_x_sq_5
       sigma_y_sq_5 = mu_4_5(Y_2_5)-mu_y_sq_5
       sigma_xy_5 = mu_5_5(XY_5)-mu_xy_5
       A1_5 = mu_xy_5*2+ 0.0001 
       A2_5 = sigma_xy_5*2+ 0.0009  
       B1_5  = mu_x_sq_5+mu_y_sq_5+ 0.0001 
       B2_5 = sigma_x_sq_5+sigma_y_sq_5+ 0.0009 
       A_5 = cm_7_5({A1_5,A2_5})
       B_5 = cm_8_5({B1_5,B2_5})
       ssim_array_5 = torch.mean(cd_1_5({A_5,B_5}))
        mcs_array_5 = torch.mean(cd_2_5({A2_5,B2_5}))
      

      ssim_array_5= (1+ssim_array_5)/2
      mcs_array_1=(1+mcs_array_1)/2
      mcs_array_2=(1+mcs_array_2)/2
      mcs_array_3=(1+mcs_array_3)/2
      mcs_array_4=(1+mcs_array_4)/2

pro1=(mcs_array_1^0.0448)*(mcs_array_2^0.2856)*(mcs_array_3^0.3001)*(mcs_array_4^0.2363)
pro2=ssim_array_5 ^0.1333
overall_mssim = -pro1*pro2

   return overall_mssim
end

df_MSSSIM = grad(MSssim) 

-------------------L1-L2------------------------
delta=0.05
--2cu-b^2
cmH1 =  grad.nn.CMulTable()
cmH2 =  grad.nn.CMulTable()
absval=grad.nn.Abs()

L1_L2_func = function (y_imH,x_imH)
	im1H = y_imH;
	im2H = x_imH;
	diff= im1H-im2H
	diffabs = absval(diff)                                                                     	
	sqmin= torch.cmin(diffabs, delta)--b
	sqterm = cmH1({sqmin,sqmin})--b^2
	c = torch.cmin(diffabs, delta)
	mk=torch.eq(c,delta)
	c[mk]=1
	diffabs_c = cmH2({diffabs,c})
	h1=0.5 * (2*diffabs_c - sqterm)
	L1_L2_loss= torch.mean(h1)
	return L1_L2_loss
end

df_HUBER = grad(L1_L2_func) 

local fake

local fDx = function(x)
   netD:apply(function(m) 
                if torch.type(m):find('Convolution') then 
                  m.bias:zero() 
                end 
              end)
   netG:apply(function(m) 
                if torch.type(m):find('Convolution') then 
                  m.bias:zero() 
                end 
              end)

   gradParametersD:zero()

	local real_ctx2ch = data:getBatch() 
	local real_ctx = real_ctx2ch[{{},{1},{},{}}]:clone() 
	real_center= real_ctx2ch[{{},{1},{},{}}]:clone() 
	input_center=real_center:clone() 
	input_real_center=real_center:clone() 
	mask = real_ctx2ch[{{},{2},{},{}}]:clone()
	mask=mask:byte()  
	real_ctx[{{},{1},{},{}}][mask] = -0.5  
	input_ctx= real_ctx:clone() 
	label:fill(real_label) 
	local output
	   
	input_center=input_center:cuda()
	output = netD:forward(input_center) 
	errD_real = criterion:forward(output, label) 
	errD_realTotal = errD_realTotal + ( errD_real * (real_ctx:size(1)) )

	local df_do = criterion:backward(output, label) 
	netD:backward(input_center, df_do)
	input_ctx=input_ctx:cuda()
	fake = netG:forward(input_ctx)  
	input_center=fake:clone()
	label:fill(fake_label)
	local output
	output = netD:forward(input_center)
	errD_fake = criterion:forward(output, label)
	errD_fakeTotal = errD_fakeTotal + (errD_fake * (real_ctx:size(1) ))
	local df_do = criterion:backward(output, label)
	netD:backward(input_center, df_do)
	errD = errD_real + errD_fake
	counterD = counterD + real_ctx:size(1) 
	return errD, gradParametersD
end

 
local fGx = function(x)
   netD:apply(function(m) 
               if torch.type(m):find('Convolution') then 
                 m.bias:zero() 
               end
              end)
            
   netG:apply(function(m) 
               if torch.type(m):find('Convolution') then 
                 m.bias:zero() 
               end 
              end)

   gradParametersG:zero()

   label:fill(real_label)


   local output = netD.output 
   errG = criterion:forward(output, label)
   errG_adv_EpochAvg = errG_adv_EpochAvg + (errG * (input_center:size(1)) )
   local df_do = criterion:backward(output, label)
   local df_dg
   df_dg = netD:updateGradInput(input_center, df_do)
   local df_dg_MSssim
   df_dg_MSssim = df_MSSSIM(fake:float(),input_real_center:float())
   df_dg_MSssim=df_dg_MSssim:cuda()
   fakesv=image.toDisplayTensor{input=fake[1]:float()}; image.save('fake.png',fakesv)
   realsv=image.toDisplayTensor{input=input_real_center[1]:float()}; image.save('real.png',realsv)
   masksv=image.toDisplayTensor{input=input_ctx[1]:float()}; image.save('masked.png',masksv)
   local df_dg_huber
   df_dg_huber = df_HUBER(fake:float(),input_real_center:float())
   df_dg_huber=df_dg_huber:cuda()
   input_real_center=input_real_center:cuda()
   
   MSssim_error=0
  
	for i = 1,real_center:size(1) do
		local df_dummy,MSssim_err

		df_dummy, MSssim_err = df_MSSSIM((fake[i][{{},{},{}}]):float(),(input_real_center[i][{{},{},{}}]):float())	   
		MSssim_error = MSssim_error + MSssim_err
		MSssim_err=nil
		df_dummy=nil
		collectgarbage()
	end 
    MSssim_error = MSssim_error/real_center:size(1)
   errG_msssim_EpochAvg=errG_msssim_EpochAvg + (MSssim_error * (input_center:size(1)) )
	errG_huber=0

	 for i = 1,real_center:size(1) do
		 local df_dummyH,huber_err
			df_dummyH, huber_err = df_HUBER((fake[i][{{},{},{}}]):float(),(input_real_center[i][{{},{},{}}]):float())	   
			errG_huber = errG_huber + huber_err
			
			collectgarbage()
		end 

    errG_huber = errG_huber/real_center:size(1)
	errG_huber_EpochAvg=errG_huber_EpochAvg + (errG_huber * (input_center:size(1)) )
	errG_total = (0.4995 * (1+MSssim_error)) + (0.4995 * errG_huber) + (0.001*errG)
	errG_total_EpochAvg=errG_total_EpochAvg + (errG_total * (input_center:size(1)))
    df_dg_MSssim:mul(0.4995):add(0.4995, df_dg_huber)
    df_dg_MSssim:add(0.001,df_dg)
    netG:backward(input_ctx, df_dg_MSssim)
   
    counterG = counterG + input_center:size(1) 
   return errG_total, gradParametersG
end

local logger1 = optim.Logger('Train.log')
logger1:setNames{'ErrDR(BCE_RealLbl_RealIpToD)', 'ErrDF(BCE_FakeLbl_FakeIpToD)', 'errG_huber', 'errG_msssim', 'errG_adv(BCE_RealLbl_FakeIpToD)', 'ErrG_total'}

 
---------------------------------------------------------------------------
-- Train 
---------------------------------------------------------------------------
for epoch = 1, opt.niter do  

  errG_total_EpochAvg=0
  errG_huber_EpochAvg=0
  errG_adv_EpochAvg=0
  errG_msssim_EpochAvg=0
  errD_realTotal=0
  errD_fakeTotal=0
  counterD=0
  counterG=0
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do 
	optim.sgd(fDx, parametersD, optimStateD)
	optim.adam(fGx, parametersG, optimStateG)
	if errG < 0.001 then
		optim.sgd(fDx, parametersD, optimStateD)
		optim.sgd(fDx, parametersD, optimStateD)
	end
    if ( errD_fake ==0 ) or ( errD_real == 0 ) then
 
		optim.adam(fGx, parametersG, optimStateG)
		optim.adam(fGx, parametersG, optimStateG)
		optim.adam(fGx, parametersG, optimStateG)
		optim.adam(fGx, parametersG, optimStateG)
        optim.adam(fGx, parametersG, optimStateG)
	end
	if ( errD_fake < 0.005 and errD_fake ~=0)  or ( errD_real < 0.001 and errD_real ~=0 ) then
	  optim.adam(fGx, parametersG, optimStateG)
	end
		
	print(('Epoch:[%d][%5d/%5d]\t'
						       .. 'ErrDR(BCE_RealLbl_RealIpToD): %.4f  ErrDF(BCE_FakeLbl_FakeIpToD): %.4f  Err_G(BCE_RealLbl_FakeIpToD): %.4f   MSSSIM: %.4f L1SOFT: %.4f errG_total: %.4f  '):format(
						     epoch, 
                             ((i-1) / opt.batchSize),  
						     math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),  
						    errD_real, errD_fake,
                            
						     errG ,
                             -MSssim_error,
                             errG_huber,
                             errG_total )) 
             



   end  
   
  
   errG_huber_EpochAvg = errG_huber_EpochAvg/counterG
   errG_msssim_EpochAvg = errG_msssim_EpochAvg/counterG
   errG_adv_EpochAvg = errG_adv_EpochAvg/counterG
   errG_total_EpochAvg = errG_total_EpochAvg/counterG
   errD_realTotal=errD_realTotal/counterD
   errD_fakeTotal=errD_fakeTotal/counterD
   logger1:add{errD_realTotal, errD_fakeTotal, errG_huber_EpochAvg, errG_msssim_EpochAvg , errG_adv_EpochAvg,  errG_total_EpochAvg }
   parametersD, gradParametersD = nil, nil  
   parametersG, gradParametersG = nil, nil
   util.save(string.format("netG_%d.t7", epoch), netG, opt.gpu)
   util.save(string.format(" netD_%d.t7", epoch), netD, opt.gpu)

   parametersD, gradParametersD = netD:getParameters()  
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d'):format(
            epoch, opt.niter))
optimStateG.learningRate = optimStateG.learningRate/1.8
optimStateD.learningRate = optimStateD.learningRate/1.8
end 
