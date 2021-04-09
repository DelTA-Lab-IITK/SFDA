-- Torch Implementation of Domain Impression: A Source Data Free Domain Adaptation Method WACV 2021
--- Written By Vinod Kumar Kurmi (vinodkumarkurmi@gmail.com)
-- Some parts of code are taken from  https://github.com/soumith/dcgan.torch
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'optim'
require 'gnuplot'
require 'loadcaffe'
require 'image';
require 'torch';
require 'nn';
require 'xlua'
require 'loadcaffe'
require 'cudnn'
require 'dataloader/dataset-mnist'
require 'dataloader/dataset-mnistM'
require '../../../../../../NNLR/misc/nnlr/nnlr' --- for layer wise learnig rate
local c = require 'trepl.colorize'
LogSumExp = require 'LogSumExp';

-- LogSumExp=LogSumExp()



opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 33,
   fineSize = 32,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 10000,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 0,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'Logfiles',
   noise = 'normal',       -- uniform / normal
   epoch_save_modulo = 1;
   manual_seed=4,          -- Seed
  batchSize = 64,         -- batch Size
  nc = 3,                 -- # of channels in input
  save='logs/',            -- Saving the logs of trainining
  DataSet='dataset_direct',
  --momentum
    lamda=1,                -- Lamda value for gradeint reversal value.(fix)
  baseLearningRate=0.0002,
  max_epoch=10000,
  gamma=0.001,   -- for inverse policy :  base_lr * (1 + gamma * iter) ^ (- power)
  power=0.75,    -- for inverse policy :  base_lr * (1 + gamma * iter) ^ (- power)
  max_epoch_grl=10000, -- For progress in process , calculate the lamda for grl
  alpha=10,  -- LR schdular (2nd way)
}

train_gen_epoch=25

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
-- if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- Adaptation part dataset

cutorch.manualSeed(opt.manual_seed)
  torch.manualSeed(opt.manual_seed)

--=====================Tuning Parameters===================================
  local prev_accuracy=0
  batchSize =opt.batchSize
  opt.save=opt.save .. 'batchsize_' .. opt.batchSize
  torch.setnumthreads(1)
  torch.setdefaulttensortype('torch.FloatTensor')
--==============Ploting Fuction=============================================================================
  confusion = optim.ConfusionMatrix({'0','1','2','3','4','5','6','7','8','9'})
geometry = {32,32}
  print('Will save at '..opt.save)
  paths.mkdir(opt.save)
  testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
  testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
  testLogger.showPlot = false
  errorlog = optim.Logger(paths.concat(opt.save, 'error.log'))
  errorlog:setNames{'% Training Error (train set)', '% Testing Error(test set)'}
  errorlog.showPlot = false

  train_gen_error_log = optim.Logger(paths.concat(opt.save, 'train_gen_error.log'))
  train_gen_error_log:setNames{'% Train Gen Error'}
  train_gen_error_log.showPlot = false

  train_pred_error_log = optim.Logger(paths.concat(opt.save, 'train_pred_error.log'))
  train_pred_error_log:setNames{'% Train Pred Error'}
  train_pred_error_log.showPlot = false

  train_grad_log = optim.Logger(paths.concat(opt.save, 'train_grad.log'))
  train_grad_log:setNames{'% Train Grad'}
  train_grad_log.showPlot = false




 -- Target Dataset----
  -- create training set and normalize
  Num_Train_Target =59001
  Num_Test_Target = 10001
  local TargettrainPath='../../../../../../Datasets/mnist_m/mnist_m_t7_file/mnistM_train_datset.t7'
  local TargettestPath='../../../../../../Datasets/mnist_m/mnist_m_t7_file/mnistM_test_datset.t7'
  TargetTrainData = mnistM.loadTrainSet(TargettrainPath,Num_Train_Target, geometry)
  TargetTrainData:normalizeGlobal(mean, std)
  -- create test set and normalize
  TargetTestData = mnistM.loadTestSet(TargettestPath,Num_Test_Target, geometry)
  TargetTestData:normalizeGlobal(mean, std)




----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

--===FUNCTIONS==============
  function check_accuracy(scores, targets)
    local num_test = (#targets)[1]
    local no_correct = 0
    local confidences, indices = torch.sort(scores, true)
    local predicted_classes = indices[{{},{1}}]:long()
    targets = targets:long()
    no_correct = no_correct + ((torch.squeeze(predicted_classes):eq(targets)):sum())
    local accuracy = no_correct / num_test
    return accuracy
  end

function normalizeGlobal(data)
      local std =  data:std()
      local mean =  data:mean()
      data:add(-mean)
      data:mul(1/std)
      return data
   end


  function check_accuracyTest(scores, targets)
    local num_test = (#targets)[1]
    local no_correct = 0
    local confidences, indices = torch.sort(scores, true)
    local predicted_classes = indices[{{},{1}}]:long()
    targets = targets:long()
    no_correct = no_correct + ((torch.squeeze(predicted_classes):eq(targets)):sum())
    local accuracy = no_correct
    return accuracy
  end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

function ints_to_one_hot(indd, width)
    -- local height = ints:size()
    local zeros = torch.zeros(opt.batchSize,width)

    local indices = indd:view(-1, 1):long()

    local one_hot = zeros:scatter(2, indices, 1)

    return one_hot
end


local epoch_save_modulo = opt.epoch_save_modulo
print("modulo value: ", opt.epoch_save_modulo);

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz+10, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- -- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf*2, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
netD:add(SpatialConvolution(ndf , ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)


-- Doamin Adaptation model
model = nn.Sequential()
  --------Map table for two stream input(one for source data another for target data)---------------
  net1= nn.MapTable()
  net2= nn.MapTable()
  netDG= nn.MapTable()

  net11= nn.Sequential()
  net22= nn.Sequential()
  netDDG= nn.Sequential()

   ------------------------------------------------------------
      -- convolutional network
    ------------------------------------------------------------
    net11:add(nn.SpatialConvolutionMM(3, 32, 5, 5))
    net11:add(nn.ReLU(true))
    net11:add(nn.SpatialMaxPooling(2, 2, 2,2))
    net11:add(nn.SpatialConvolutionMM(32, 48, 5, 5))
    net11:add(nn.ReLU(true))
    net11:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    net11:add(nn.Reshape(48*5*5))
    -- stage 2 : standard 2-layer MLP:
    net22:add(nn.Linear(48*5*5, 100))
    net22:add(nn.ReLU(true))
    net22:add(nn.Linear(100, 100))
    net22:add(nn.ReLU(true))
    net22:add(nn.Linear(100, 10))
    net22:add(nn.LogSoftMax())
    -- Gradient Reversal Domain classifier Network
    module = nn.GradientReversal(lambda)
    netDDG:add(module)
    netDDG:add( nn.Linear( 48*5*5, 100)):learningRate('weight', 10)
                      :learningRate('bias', 20)
    netDDG:add(nn.ReLU(true))
    --netDD:add(nn.Dropout(0.5))
    netDDG:add( nn.Linear( 100, 2)):learningRate('weight', 10)
                      :learningRate('bias', 20)

    -- Map Tabel for two input----
    net1:add(net11)
    net2:add(net22)
    netDG:add(netDDG)

      --Initially Lamda set =0
    module:setLambda(0)




--============ Criterion=================
  local criterionTrain = nn.ClassNLLCriterion()
  local criterionTest = nn.ClassNLLCriterion()
  local criterionCrossE = nn.CrossEntropyCriterion()
  local criterionCrossE_parallel = nn.ParallelCriterion():add(criterionCrossE,0.1):add(criterionCrossE,0.1)
  local long_sum=nn.LogSumExp()


if opt.gpu >=0 then
    net1:cuda()
    net2:cuda()
    netDG:cuda()
    criterionTest:cuda()
    criterionTrain:cuda()
    criterionCrossE_parallel:cuda()
    long_sum:cuda()
   end


--== Different Learning rate for weigth and bias
  local temp_baseWeightDecay=0.001  --no meaningin my case
  local learningRates_Net1, weightDecays_Net1 = net1:getOptimConfig(opt.baseLearningRate,temp_baseWeightDecay)
  local learningRates_Net2, weightDecays_Net2 = net2:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
  local learningRates_NetDG, weightDecays_NetDG = netDG:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
--===========Parameters===================================
  parameters1, gradParameters1 = net1:getParameters()
  parameters2, gradParameters2 = net2:getParameters()
  parametersDG, gradParametersDG = netDG:getParameters()

  local method = 'xavier'
  net1 = require('misc/weight-init')(net1, method)
  net2 = require('misc/weight-init')(net2, method)
  netDG = require('misc/weight-init')(netDG, method)
local updated_learningrate=opt.baseLearningRate

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()




model_path='../../../pretrained_model'

  pretrained_classifier=nn.Sequential()
  pretrained_classifier:add(torch.load('pretrained_model/Accuracy0.58360182201978net1_606.t7'))
  pretrained_classifier:add(torch.load('pretrained_model/Accuracy0.58360182201978net2_606.t7'))
  logsoftmax=nn.LogSoftMax()
  logsoftmax=logsoftmax:cuda()
  logsoftmax2=nn.LogSoftMax()
  logsoftmax2=logsoftmax2:cuda()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
local parametersC, gradParametersC = pretrained_classifier:getParameters()


local criterionCross = nn.CrossEntropyCriterion()
criterionCross=criterionCross:cuda()

local criterionCrossD = nn.CrossEntropyCriterion()
criterionCrossD=criterionCrossD:cuda()

-- if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end


-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()


   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output:cuda(), label:cuda())
   local df_do = criterion:backward(output:cuda(), label:cuda())
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   noise_with_label=torch.cat(noise, onehot_class_label, 2)
   local fake = netG:forward(noise_with_label)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   gradParametersC:zero()
   gradParametersD:zero()

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   noise_with_label=torch.cat(noise, onehot_class_label, 2)
   local fake = netG:forward(noise_with_label)
   input:copy(fake)
   local output = netD:forward(input)



   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   -- local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   local class_output=pretrained_classifier:forward({input})
   local one_tensor_cls = torch.CudaTensor(class_output[1]:size()):fill(1)
   -- local one_tensor_diver = torch.CudaTensor(opt.batchSize,1):fill(1)
   -- local log_prob_cls = logsoftmax:forward(class_output[1])
   -- local grads_entropy_cls = torch.add(log_prob_cls:cuda(), one_tensor_cls:cuda())/opt.batchSize
   -- local w= torch.div(torch.sum(class_output[1], 2),opt.batchSize)
   -- local log_w=logsoftmax:forward(w)
   -- local grads_diversity_cls = -torch.add(log_w:cuda(), one_tensor_diver:cuda())
   -- local grads_diversity_cls_repeat=torch.repeatTensor(grads_diversity_cls,1,class_output[1]:size()[2]):cuda()

    cross_loss=criterionCross:forward(class_output[1],given_label)
   local dcross_loss=criterionCross:backward(class_output[1],given_label)



   local log_sum_Exp=long_sum:forward(class_output[1])
   local d_log_sum_Exp=long_sum:backward(class_output[1],one_tensor_cls)
   local exp_over_model=torch.mean(d_log_sum_Exp, 1)
   local repeat_ten=torch.expandAs(exp_over_model, d_log_sum_Exp)
   local d_data_dis=repeat_ten-d_log_sum_Exp




   local d_class_ent=pretrained_classifier:backward({input},{dcross_loss+d_data_dis})

   netG:backward(noise_with_label, alpha*df_dg+beta*d_class_ent[1])
   avg_gen_er=avg_gen_er+errG
   avg_pre_er=avg_pre_er+cross_loss
   avg_data_grd=avg_data_grd+log_sum_Exp
   return errG, gradParametersG
end

function train(epoch)
-- for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
  net1:training()
  net2:training()
  netD:training()
  epoch = epoch or 1
  if(epoch>1) then
    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
    local p=epoch/opt.max_epoch_grl
    local baseWeightDecay = torch.pow((1 +  epoch * opt.gamma), (-1  * opt.power)) -- need to chanage
    updated_learningrate=opt.baseLearningRate*baseWeightDecay
    print('Learnig Rate',updated_learningrate)
    -- lamda=(2*torch.pow(1+torch.exp(-10*p),-1))-1
    print('Lamda',opt.lamda)
    module:setLambda(opt.lamda)
  end
  local avg_loss=0
  local avg_acc=0
  local count =0
  avg_gen_er=0
   avg_pre_er=0
   avg_data_grd=0
   for t = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        data_tm:reset(); data_tm:resume()
        real = data:getBatch()
        data_tm:stop()
        given_label = torch.Tensor(opt.batchSize)
        given_label:random(1,10)
        onehot_class_label=ints_to_one_hot(given_label,10)
        onehot_class_label:view(opt.batchSize,10,1,1)
        onehot_class_label=onehot_class_label:cuda()
        given_label=given_label:cuda()


        local p=epoch/100
        if p > 2 then
          p=2
        end
        alpha=1
        beta=(2*torch.pow(1+torch.exp(-10*p),-1))-1

        -- if epoch>400 then
        --   alpha=0
        --   beta=0
        -- end
        -- -- print('alpha beta',alpha,beta)
        optim.adam(fDx, parametersD, optimStateD)
        optim.adam(fGx, parametersG, optimStateG)

        counter = counter + 1
        Targetbatch=real
        --------------------------------------------------------------------------

        if opt.noise == 'uniform' then -- regenerate random noise
          noise:uniform(-1, 1)
        elseif opt.noise == 'normal' then
          noise:normal(0, 1)
        end
      if epoch>train_gen_epoch then
        noise_with_label=torch.cat(noise, onehot_class_label, 2)
        local fake = netG:forward(noise_with_label)
        -- fake=normalizeGlobal(fake)
        -- forwardNetwork
        Targetbatch=Targetbatch:cuda()
        Targetbatch=Targetbatch:add(1):mul(0.5)
        fake=fake:add(1):mul(0.5)
        -- print('fake',fake:max(),fake:min())
        -- print('Targetbatch',Targetbatch:max(),Targetbatch:min())
        Targetbatch=normalizeGlobal(Targetbatch)
        fake=normalizeGlobal(fake)
        outputs1 = net1:forward({fake,Targetbatch})

        outputs = net2:forward(outputs1)
        outputsDG = netDG:forward(outputs1)
        outputsDG[1]=outputsDG[1]:cuda()
        outputsDG[2]=outputsDG[2]:cuda()
        local TargetDomainlable=torch.Tensor(Targetbatch:size()[1]):fill(2)
        local SourceDomainlable=torch.Tensor(fake:size()[1]):fill(1)
        if opt.gpu >=0 then
          TargetDomainlable=TargetDomainlable:cuda()
          SourceDomainlable=SourceDomainlable:cuda()
        end

        err = criterionTrain:forward(outputs[1], given_label:cuda())
        errDomain = criterionCrossE_parallel:forward(outputsDG, {SourceDomainlable,TargetDomainlable})


        gradParametersDG:zero()
        gradParameters2:zero()
        gradParameters1:zero()

        local dgradOutputsS=torch.CudaTensor()    --Declaration of dgradOutputsS for source class
        dgradOutputsS:resize(outputs[1]:size())
        dgradOutputsS:zero()
        dgradOutputsS = criterionTrain:backward(outputs[1], given_label)

          local zeros = torch.CudaTensor()     -- Zero gradient for Target data Classification(we dont have target label)
        zeros:resize(dgradOutputsS:size())
        zeros:zero()
        dgradOutputs={dgradOutputsS, zeros}

      ---- Optimization Net4-------
        feval_net2 = function(x)
        dgradOutputs_mod2 = net2:backward(outputs1, dgradOutputs)
          return err, gradParameters2
        end
        optim.sgd(feval_net2, parameters2, {
                         learningRates = learningRates_Net2,
                         weightDecays = weightDecays_Net2,
                         learningRate = updated_learningrate,
                         momentum = opt.momentum,
                        })


          dgradOutputsDomain = criterionCrossE_parallel:backward(outputsDG, {SourceDomainlable,TargetDomainlable})    -- classification loss grad

      ---- Optimization Domain Confusion Branch -------
        feval_netDG = function(x)
          dgradOutputs_modDG  = netDG:backward(outputs1, dgradOutputsDomain)
          return err, gradParametersDG
        end
        optim.sgd(feval_netDG, parametersDG, {
                         learningRates = learningRates_NetDG,
                         weightDecays = weightDecays_NetDG,
                         learningRate = updated_learningrate,
                         momentum = opt.momentum,
                        })
      ---- Optimization netB(bottleneck_ Branch -------
        local total_grad={}
        total_grad[1] = dgradOutputs_mod2[1]+ dgradOutputs_modDG[1]
        total_grad[2] = dgradOutputs_mod2[2]+ dgradOutputs_modDG[2]


        feval_net1 = function(x)
          dgradOutputs_mod1   = net1:backward({fake,Targetbatch},total_grad)
          return err, gradParameters1
        end
        optim.sgd(feval_net1, parameters1, {
                         learningRates = learningRates_Net1,
                         weightDecays = weightDecays_Net1,
                         learningRate = updated_learningrate,
                         momentum = opt.momentum,
                        })


      local train_acc = check_accuracy(outputs[1], given_label)
      avg_loss=avg_loss+err
      avg_acc=avg_acc+train_acc
      train_acc =nil
      err=nil
      count=count+1

      -- -- logging
      --   if ((t-1) / opt.batchSize) % 1 == 0 then
      --      print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
      --                .. '  Err_G: %.4f  Err_D: %.4f Err_C: %.4f'):format(
      --              epoch, ((t-1) / opt.batchSize),
      --              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
      --              tm:time().real, data_tm:time().real,
      --              errG and errG or -1, errD and errD or -1,cross_loss))
      --   end

    end



   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   if epoch % epoch_save_modulo == 0 then -- allows to pass in modulo value to only save checkpoints at certain intervals
      torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
      torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
    end
      parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
      parametersG, gradParametersG = netG:getParameters()
      print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
               epoch, opt.niter, epoch_tm:time().real))

    return avg_acc/count,  avg_loss/count, avg_gen_er/count, avg_pre_er/count,avg_data_grd/count
-- end
end
function test(epoch)
  -- disable flips, dropouts and batch normalization
  net1:evaluate()
  net2:evaluate()
  netD:evaluate()
  local err_val=0
  local avg_test_acc=0
  local count=0
  for t = 1, TargetTestData:size(), opt.batchSize do
    local TargetTestbatch = torch.Tensor(math.min(t+opt.batchSize-1,TargetTestData:size())-t+1,3,geometry[1],geometry[2])
    local TargetTestbatchLabel = torch.Tensor(math.min(t+opt.batchSize-1,TargetTestData:size())-t+1)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,TargetTestData:size()) do
      local sample = TargetTestData[i]
      local InputsTargetTest = sample[1]:clone()
      local _,LabelTargetTest = sample[2]:clone():max(1)
      LabelTargetTest = LabelTargetTest:squeeze()
      TargetTestbatch[k] = InputsTargetTest
      TargetTestbatchLabel[k] = LabelTargetTest
      k = k + 1
    end

    if opt.gpu >=0 then
      TargetTestbatch=TargetTestbatch:cuda()
      TargetTestbatchLabel=TargetTestbatchLabel:cuda()
    end
    local outputs1 = net1:forward({TargetTestbatch})
    local outputs = net2:forward(outputs1)
    confusion:batchAdd(outputs[1], TargetTestbatchLabel)
    err_val = err_val+ criterionTest:forward(outputs[1], TargetTestbatchLabel)  -- Classification Loss
    count=count+1
    local test_batch_acc = check_accuracyTest(outputs[1], TargetTestbatchLabel)
    avg_test_acc=avg_test_acc+test_batch_acc
    test_batch_acc=nil

  end
  confusion:updateValids()
  test_accuracy=confusion.totalValid
  if not testLogger then
    confusion:zero()
  end
   return err_val/count, test_accuracy,avg_test_acc/TargetTestData:size()
end

function save_html(train_acc,test_acc,train_err,test_err,train_gen_error,train_pred_error,train_grad,epoch)
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, test_acc}
    testLogger:style{'-','-'}
    -- testLogger:plot()
    errorlog:add{train_err, test_err}
    errorlog:style{'-','-'}

    train_gen_error_log:add{train_gen_error}
    train_gen_error_log:style{'-'}

    train_pred_error_log:add{train_pred_error}
    train_pred_error_log:style{'-'}

    -- train_grad_log:add{train_grad:max(),train_grad:min()}
    -- train_grad_log:style{'-','-'}


    -- errorlog:plot()
    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      -- do
      --   os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      --   os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      --   local f = io.open(opt.save..'/test.base64')
      --   if f then base64im = f:read'*all' end
      -- end
      -- local base64im_error
      -- do
      --   os.execute(('convert -density 200 %s/error.log.eps %s/error.png'):format(opt.save,opt.save))
      --   os.execute(('openssl base64 -in %s/error.png -out %s/error.base64'):format(opt.save,opt.save))
      --   local f = io.open(opt.save..'/error.base64')
      --   if f then base64im_error = f:read'*all' end
      -- end
      local file = io.open(opt.save..'/report.html','w')
      -- file:write('<h5>Training Source data size:  '..SourceTrainData:size()..'\n')
      file:write('<h5>Training Target data size:  '..TargetTrainData:size()..'\n')
      -- file:write('<h5> Source test data size:  '..SourceTestData:size()..'\n')
      file:write('<h5> Target test data size:  '..TargetTestData:size()..'\n')
      file:write('<h5>batchSize:  '..batchSize..'\n')
      file:write('<h5>Base Learning Rate:  '..opt.baseLearningRate..'\n')
      file:write('<h5>Seed :  '..opt.manual_seed..'\n')
      file:write('<h5>lamda :  '..opt.lamda..'\n')
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write'</pre></body></html>'
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <table>
      ]]):format(opt.save,epoch,base64im))
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <table>
      ]]):format(opt.save,epoch,base64im_error))

      file:close()
    end
    confusion:zero()
  end

    --print('epoch',epoch)
          if prev_accuracy< test_acc then
    print('Model is saving')
    collectgarbage()
        net1:clearState()
        net2:clearState()
        netD:clearState()
    torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'net1_' .. epoch .. '.t7'),net1) -- defined in util.lua
    torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'net2_' .. epoch .. '.t7'),net2)
    torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'netD_' .. epoch .. '.t7'),netD)
    print('Model is Saved')
    prev_accuracy=test_acc
    end
end

for i=1,opt.niter do
train_acc,train_loss,train_gen_error,train_pred_error,train_grad=train(i)
 if i>train_gen_epoch then
  print('Train_acc',train_acc,'Train_loss',train_loss)
  -- collectgarbage()
  test_loss,test_acc,test_acc_2=test(i)
  print('test_acc',test_acc, 'test_acc_2',test_acc_2,'Test_loss',test_loss)
  save_html(train_acc,test_acc,train_loss,test_loss, train_gen_error,train_pred_error,train_grad,i)
  -- collectgarbage()
  end
end
