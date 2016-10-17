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
cmd:option('-hflip', 0, 'proba horizontal flip')
cmd:option('-vflip', 0, 'proba vertical flip')
cmd:option('-model', 'resnet', 'or vgg16')
cmd:option('-pathnet', 'logs/m2caiworkflow/resnet/16_10_16_15:01:49/net.t7', '')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unistd.getpid()

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '../Deep6Framework2'
local pathmodel = path..'/models/raw/'..config.model..'/net.t7'
local pathdataset = path..'/data/processed/m2caiworkflow'
local pathtrainset = pathdataset..'/trainset.t7'
local pathclasses  = pathdataset..'/classes.t7'
local pathclass2target = pathdataset..'/class2target.t7'
local pathnet = path..'/'..config.pathnet
os.execute('mkdir -p '..pathdataset)

local trainset, classes, class2target = m2caiworkflow.loadFullTrainset()

local model = vision.models[config.model]
require 'cudnn'
local net = torch.load(pathnet)
print(net)
local criterion = nn.CrossEntropyCriterion():float()

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

trainset = trainset:shuffle()--(300)
-- valset   = valset:shuffle(300)
trainset = addTransforms(trainset, model)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
torch.save(pathtrainset, trainset)

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
   avgvm = tnt.AverageValueMeter(),
   confm = tnt.ConfusionMeter{k=#classes},
   timem = tnt.TimeMeter{unit = false},
   clerr = tnt.ClassErrorMeter{topk = {1,5}}
}

local function createLog(mode, pathlog)
   local keys = {'epoch', 'loss', 'acc1', 'acc5', 'time'}
   local format = {'%d', '%10.5f', '%3.2f', '%3.2f', '%.1f'}
   local log = tnt.Log{
      keys = keys,
      onFlush = {
         logtext{filename=pathlog, keys=keys},
         logtext{keys=keys, format=format},
      },
      onSet = {
         logstatus{filename=pathlog},
         logstatus{}, -- print status to screen
      }
   }
   log:status("Mode "..mode)
   return log
end
local log = {
   train = createLog('train', pathtrainlog)
}

local engine = tnt.OptimEngine()
engine.epoch=1
engine.hooks.onStart = function(state)
   for _,m in pairs(meter) do m:reset() end
end
-- engine.hooks.onStartEpoch = function(state) -- training only
--    engine.epoch = engine.epoch and (engine.epoch + 1) or 1
-- end
engine.hooks.onForwardCriterion = function(state)
   meter.timem:incUnit()
   meter.avgvm:add(state.criterion.output)
   meter.clerr:add(state.network.output, state.sample.target)
   meter.confm:add(state.network.output, state.sample.target)
   log[engine.mode]:set{
      epoch = engine.epoch,
      loss  = meter.avgvm:value(),
      acc1  = 100 - meter.clerr:value{k = 1},
      acc5  = 100 - meter.clerr:value{k = 5},
      time  = meter.timem:value()
   }
   print(string.format('%s; avg. loss: %2.4f; avg. error: %2.4f',
      engine.mode, meter.avgvm:value(), meter.clerr:value{k = 1}))
end
engine.hooks.onEnd = function(state)
   print('End of epoch '..engine.epoch..' on '..engine.mode..'set')
   log[engine.mode]:flush()
   print('Confusion Matrix (rows = gt, cols = pred)')
   print(meter.confm:value())
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

-- Iterator
local trainiter = getIterator('train')

local bestepoch = {
   clerrtop1 = 100,
   clerrtop5 = 100,
   epoch = 0
}

print('Testing ...')
engine.mode = 'train'
engine:test{
   network   = net,
   iterator  = trainiter,
   criterion = criterion,
}
