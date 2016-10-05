local tnt = require 'torchnet'
local vision = require 'torchnet-vision'
require 'image'
require 'os'
require 'optim'
ffi = require 'ffi'
unistd = require 'posix.unistd'
local lsplit    = string.split
local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'
local utils     = require 'src.data.utils'
local m2caiworkflow = require 'src.data.m2caiworkflow'

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 15, 'batch size')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
cmd:option('-pathfeats', 'features/m2caiworkflow/'..os.date("%y_%m_%d_%X"), '')
cmd:option('-hflip', 0, 'proba horizontal flip')
cmd:option('-vflip', 0, 'proba vertical flip')
cmd:option('-model', 'inceptionv3', 'or vgg16')
cmd:option('-layerid', 30, 'vgg16:33,34,36,37')
cmd:option('-nfeats', 2048, 'features number (or vgg16:4096)')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unistd.getpid()

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '.'
local pathmodel = path..'/models/raw/'..config.model..'/net.t7'
local pathdataset = path..'/data/processed/m2caiworkflow'
local pathtrainset = pathdataset..'/trainset.t7'
local pathvalset   = pathdataset..'/valset.t7'
local pathclasses  = pathdataset..'/classes.t7'
local pathclass2target = pathdataset..'/class2target.t7'
os.execute('mkdir -p '..pathdataset)

local trainset, valset, classes, class2target = m2caiworkflow.load()

local model = vision.models[config.model]
local net = model.loadExtracting{
   filename = pathmodel,
   layerid  = config.layerid
}
print(net)

local pathfeats = path..'/'..config.pathfeats
local pathconfig = pathfeats..'/config.t7'
os.execute('mkdir -p '..pathfeats)
torch.save(pathconfig, config)

local function addTransforms(dataset, model)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.line,', ')
      sample.path   = spl[1]
      sample.target = spl[2] + 1
      sample.label  = classes[spl[2] + 1]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.randomScale{
            minSize=model.inputSize[2],
            maxSize=model.inputSize[2]+11
         },
         vision.image.transformimage.randomCrop(model.inputSize[2]),
         -- vision.image.transformimage.horizontalFlip(config.hflip),
         -- vision.image.transformimage.verticalFlip(config.vflip),
         -- vision.image.transformimage.rotation(0),
         vision.image.transformimage.colorNormalize{
            mean = model.mean,
            std  = model.std
         }
      }(sample.path)
      return sample
   end)
   return dataset
end

-- trainset = trainset:shuffle(300)--(300)
-- valset   = valset:shuffle(300)
trainset = addTransforms(trainset, model)
valset   = addTransforms(valset, model)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
torch.save(pathtrainset, trainset)
torch.save(pathvalset, valset)
torch.save(pathclasses, classes)
torch.save(pathclass2target, class2target)

local function getIterator(mode)
   -- mode = {train,val,test}
   local iterator = tnt.ParallelDatasetIterator{
      nthread   = config.nthread,
      init      = function()
         require 'torchnet'
         require 'torchnet-vision'
      end,
      closure   = function(threadid)
         local dataset = torch.load(pathdataset..'/'..mode..'set.t7')
         return dataset:batch(config.bsize)
      end,
      transform = function(sample)
         if mode ~= 'test' then
            sample.target = torch.Tensor(sample.target):view(-1,1)
         end
         return sample
      end
   }
   print('Stats of '..mode..'set')
   for i, v in pairs(iterator:exec('size')) do
      print(i, v)
   end
   return iterator
end

local meter = {
   timem = tnt.TimeMeter{unit = false},
}

local engine = tnt.OptimEngine()
local file
engine.hooks.onStart = function(state)
   for _,m in pairs(meter) do m:reset() end
   file = assert(io.open(pathfeats..'/'..engine.mode..'extract.csv', "w"))
   file:write('path;gttarget;gtclass')
   for i=1, config.nfeats do file:write(';feat'..i) end
   file:write("\n")
end
engine.hooks.onForward = function(state)
   local output = state.network.output
   print(output:size())
   for i=1, output:size(1) do
      file:write(state.sample.path[i]);      file:write(';')
      file:write(state.sample.target[i][1]); file:write(';')
      file:write(state.sample.label[i])
      for j=1, output:size(2) do
         file:write(';'); file:write(output[i][j])
      end
      file:write("\n")
   end
end
engine.hooks.onEnd = function(state)
   print('End of extracting on '..engine.mode..'set')
   print('Took '..meter.timem:value())
   file:close()
end
if config.usegpu then
   require 'cutorch'
   cutorch.manualSeed(config.seed)
   require 'cunn'
   require 'cudnn'
   cudnn.convert(net, cudnn)
   net       = net:cuda()
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

print('Training ...')
engine.mode = 'train'
engine:test{
   network  = net,
   iterator = getIterator('train')
}
print('Validating ...')
engine.mode = 'val'
engine:test{
   network  = net,
   iterator = getIterator('val')
}




-- engine.hooks.onStart = function(state)
--    for _,m in pairs(meter) do m:reset() end
--    file = assert(io.open(pathfeats..'/'..engine.mode..'extract.csv', "w"))
--    file:write('path')
--    for i=1, 2048 do file:write(';feat'..i) end
--    file:write("\n")
-- end
-- engine.hooks.onForward = function(state)
--    local output = state.network.output
--    for i=1, output:size(1) do
--       file:write(state.sample.path[i]);
--       for j=1, output:size(2) do
--          file:write(';'); file:write(output[i][j])
--       end
--       file:write("\n")
--    end
-- end
-- if config.usegpu then
--    local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
--    engine.hooks.onSample = function(state)
--       igpu:resize(state.sample.input:size()):copy(state.sample.input)
--       state.sample.input = igpu
--    end
-- end
--
-- print('Testing ...')
-- engine.mode = 'test'
-- engine:test{
--    network  = net,
--    iterator = getIterator('test')
-- }
