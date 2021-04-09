require 'image'
require 'nn'
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
-- local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 32,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 0,           -- Display image: 0 = false, 1 = true
    nz = 100,
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

function ints_to_one_hot_singleclass(ints, width)
    -- local height = ints:size()
    local zeros = torch.zeros(opt.batchSize,width)

    local indd=torch.Tensor(opt.batchSize):fill(ints)
    local indices = indd:view(-1, 1):long()

    local one_hot = zeros:scatter(2, indices, 1)

    return one_hot
end


function ints_to_one_hot(indd, width)
    -- local height = ints:size()
    local zeros = torch.zeros(opt.batchSize,width)

    -- local indd=torch.Tensor(opt.batchSize):fill(ints)
    local indices = indd:view(-1, 1):long()

    local one_hot = zeros:scatter(2, indices, 1)

    return one_hot
end


function pred_class(scores)
        local confidences, indices = torch.sort(scores, true)
        local predicted_classes = indices[{{},{1}}]:long()
        return predicted_classes
    end




-- local class_label=9
   local given_label = torch.Tensor(opt.batchSize) --:fill(class_label)
   given_label:random(1,10)

   given_label:fill(5)
   onehot_class_label=ints_to_one_hot(given_label,10)
      onehot_class_label:view(opt.batchSize,10,1,1)
      onehot_class_label=onehot_class_label:cuda()
      given_label=given_label:cuda()




--    onehot_class_label=ints_to_one_hot(class_label,10)
--       onehot_class_label:view(opt.batchSize,10,1,1)
--       onehot_class_label=onehot_class_label:cuda()
-- given_label=given_label:cuda()

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net = torch.load(opt.net)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

print(net)

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end

local sample_input = torch.randn(2,100,1,1)
if opt.gpu > 0 then
    net:cuda()
    cudnn.convert(net, cudnn)
    noise = noise:cuda()
    sample_input = sample_input:cuda()
else
   sample_input = sample_input:float()
   net:float()
end

noise_with_label=torch.cat(noise, onehot_class_label, 2)

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
-- optnet.optimizeMemory(net, sample_input)

local images = net:forward(noise_with_label)
print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor(images))
print('Saved image to: ', opt.name .. '.png')

 pretrained_classifier=nn.Sequential()
  pretrained_classifier:add(torch.load('../../../pretrained_model/net1.t7'))
  pretrained_classifier:add(torch.load('../../../pretrained_model/net2.t7'))
  images=images:cuda()
  local out_class=pretrained_classifier:forward({images})
  local predict_class=pred_class(out_class[1])
  print(predict_class)

  local criterionCross = nn.CrossEntropyCriterion()
criterionCross=criterionCross:cuda()
cross_loss=criterionCross:forward(out_class[1],given_label)
print('cross_loss',cross_loss)
-- print('given_label',given_label)
  -- print('out_class[1]',nn.SoftMax():cuda():forward(out_class[1])*100)

if opt.display then
    disp = require 'display'
    disp.image(images)
    print('Displayed image')
end
