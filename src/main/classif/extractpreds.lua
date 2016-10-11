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
local transformimage =
   require 'torchnet-vision.image.transformimage'
local m2caiworkflow = require 'src.data.m2caiworkflow'

local cmd = torch.CmdLine()
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 7, 'batch size')
cmd:option('-nthread', 3, 'threads number for parallel iterator')
cmd:option('-pathnet','logs/m2caiworkflow/finetuning_part2/16_09_13_08:35:55/net.t7')
cmd:option('-pathpreds', '/local/robert/m2cai/workflow/extract/inceptionv3_2/16_09_13_08:35:55')
cmd:option('-part', '2', '')
cmd:option('-model', 'inceptionv3', '')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unilocal tnt = require 'torchnet'
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '/net/big/cadene/doc/Deep6Framework2'
local pathdataset  = path..'/data/processed/m2caiworkflow'
local pathtrainset = pathdataset..'/trainset.t7'
local pathvalset   = pathdataset..'/valset.t7'
local pathtestset  = pathdataset..'/testset.t7'
local pathnet = config.pathnet

require 'cunn'
require 'cudnn'
local net = torch.load(pathnet)
print(net)
local criterion = nn.CrossEntropyCriterion():float()

local trainset, valset, classes, class2target = m2caiworkflow.load(config.part)
local testset = m2caiworkflow.loadTestset()
-- testset  = testset:shuffle(300)
-- valset   = valset:shuffle(300)
-- trainset = trainset:shuffle(300)

local function addTransforms(dataset, model)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.line,', ')
      sample.path   = spl[1]
      sample.target = spl[2] + 1
      sample.label  = classes[spl[2] + 1]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.randomScale{
            minSize = model.inputSize[2],
            maxSize = model.inputSize[2] + 31},
         vision.image.transformimage.randomCrop(model.inputSize[2]),
         vision.image.transformimage.colorNormalize{
            mean = model.mean,
            std  = model.std
         }
      }(sample.path)
      return sample
   end)
   return dataset
end

local model = vision.models[config.model]
testset  = addTransforms(testset, model)
valset   = addTransforms(valset, model)
trainset = addTransforms(trainset, model)
function trainset:manualSeed(seed) torch.manualSeed(seed) end

os.execute('mkdir -p '..pathdataset)
os.execute('mkdir -p '..config.pathpreds)
torch.save(pathtrainset, trainset)
torch.save(pathvalset, valset)
torch.save(pathtestset, testset)

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
         sample.target = torch.Tensor(sample.target):view(-1,1)
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
local count, nbatch
engine.hooks.onStart = function(state)
   count = 1
   nbatch = state.iterator:execSingle("size")
   for _,m in pairs(meter) do m:reset() end
   print(engine.mode)
   file = assert(io.open(config.pathpreds..'/'..engine.mode..'set.csv', "w"))
   file:write('path;gttarget;gtclass')
   for i=1, #classes do file:write(';pred'..i) end
   file:write("\n")
end
engine.hooks.onForward = function(state)
   local output = state.network.output
   print('Mode', engine.mode)
   print('inputid', count..' / '..nbatch)
   print('#input',state.sample.input:size(1))
   print('#output',output:size(1))
   print('#target',state.sample.target:size(1))
   print('#path',#state.sample.path)
   count = count + 1
   if state.sample.input:size(1) == output:size(1) then -- hotfix
      for i=1, output:size(1) do
         -- print(state.sample.path[i])
         file:write(state.sample.path[i]);
         if engine.mode ~= 'test' then
            file:write(';')
            file:write(state.sample.target[i][1]); file:write(';')
            file:write(state.sample.label[i])
         end
         for j=1, output:size(2) do
            file:write(';'); file:write(output[i][j])
         end
         file:write("\n")
      end
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
   criterion = criterion:cuda()
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

print('Extracting trainset ...')
engine.mode = 'train'
engine:test{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion
}

print('Extracting valset ...')
engine.mode = 'val'
engine:test{
   network   = net,
   iterator  = getIterator('val'),
   criterion = criterion
}

print('Extracting testset ...')
engine.mode = 'test'
engine:test{
   network   = net,
   iterator  = getIterator('test'),
   criterion = criterion
}
