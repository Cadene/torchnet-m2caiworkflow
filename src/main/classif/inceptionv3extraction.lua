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
local transformimage = require 'torchnet-vision.image.transformimage'
local utils         = require 'src.data.utils'
local m2caiworkflow = require 'src.data.m2caiworkflow'

local cmd = torch.CmdLine()
-- options to get acctop1=79.06 in 9 epoch
cmd:option('-seed', 1337, 'seed for cpu and gpu')
cmd:option('-usegpu', true, 'use gpu')
cmd:option('-bsize', 40, 'batch size')
cmd:option('-nepoch', 20, 'epoch number')
cmd:option('-lr', 1e-4, 'learning rate for adam')
cmd:option('-lrd', 0, 'learning rate decay')
cmd:option('-nthread', 5, 'threads number for parallel iterator')
cmd:option('-part', '1', 'data part split')
local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

config.idGPU = os.getenv('CUDA_VISIBLE_DEVICES') or -1
config.pid   = unistd.getpid()
config.date  = os.date("%y_%m_%d_%X")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(config.seed)

local path = '/net/big/cadene/doc/Deep6Framework2'
local pathinceptionv3 = path..'/models/raw/inceptionv3/net.t7'
local pathdataset     = path..'/data/processed/m2caiworkflow'
local pathtrainset = pathdataset..'/trainset.t7'
local pathvalset  = pathdataset..'/valset.t7'
local pathclasses  = pathdataset..'/classes.t7'
local pathmean     = pathdataset..'/mean.t7'
local pathstd      = pathdataset..'/std.t7'
local pathclass2target  = pathdataset..'/class2target.t7'

local pathlog = path..'/logs/m2caiworkflow/inceptionv3extract_part'..config.part..'/'..config.date
local pathtrainlog  = pathlog..'/trainlog.txt'
local pathvallog   = pathlog..'/vallog.txt'
local pathbestepoch = pathlog..'/bestepoch.t7'
local pathbestnet   = pathlog..'/net.t7'
local pathconfig    = pathlog..'/config.t7'

local trainset, valset, classes, class2target = m2caiworkflow.load(config.part)

local net = vision.models.inceptionv3.load{
   filename = pathinceptionv3
}
net:remove()
net:remove()
net:add(nn.GradientReversal(0))
net:add(nn.Linear(2048,8))
print(net)
--do return end
local criterion = nn.CrossEntropyCriterion():float()

local mean, std

local function addTransforms(dataset, mean, std)
   dataset = dataset:transform(function(sample)
      local spl = lsplit(sample.line,', ')
      sample.path   = spl[1]
      sample.target = spl[2] + 1
      sample.label  = classes[spl[2] + 1]
      sample.input  = tnt.transform.compose{
         function(path) return image.load(path, 3) end,
         vision.image.transformimage.randomScale{minSize=299,maxSize=330},
         vision.image.transformimage.randomCrop(299),
         vision.image.transformimage.colorNormalize(mean, std) -- mean not set
      }(sample.path)
      return sample
   end)
   return dataset
end

trainset = trainset:shuffle()
trainset = addTransforms(trainset, mean, std)
function trainset:manualSeed(seed) torch.manualSeed(seed) end
-- valset  = valset:shuffle(300)
valset  = addTransforms(valset, mean, std)

if config.fromscratch then
   net:reset()
   if paths.filep(pathmean) and paths.filep(pathstd) then
      mean = torch.load(pathmean)
      std  = torch.load(pathstd)
   else
      mean = torch.zeros(3,299,299)
      std = torch.zeros(3,299,299)
      mean, std = utils.processMeanStd(trainset, .1, mean, std)
      torch.save(pathmean, mean)
      torch.save(pathstd, std)
   end
else
   mean = vision.models.inceptionv3.mean
   std  = vision.models.inceptionv3.std
end

os.execute('mkdir -p '..pathlog)
os.execute('mkdir -p '..pathdataset)
torch.save(pathconfig, config)
torch.save(pathtrainset, trainset)
torch.save(pathvalset, valset)

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
   train = createLog('train', pathtrainlog),
   val   = createLog('val', pathvallog)
}

local engine = tnt.OptimEngine()
engine.hooks.onStart = function(state)
   for _,m in pairs(meter) do m:reset() end
end
engine.hooks.onStartEpoch = function(state) -- training only
   engine.epoch = engine.epoch and (engine.epoch + 1) or 1
end
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
   print(string.format('%s epoch: %i; avg. loss: %2.4f; avg. error: %2.4f',
      engine.mode, engine.epoch, meter.avgvm:value(), meter.clerr:value{k = 1}))
   -- local p,g=state.network:parameters()
   -- print(p[1][{1,1,1,1}])
   -- print(g[1][{1,1,1,1}])
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
local valiter  = getIterator('val')

local bestepoch = {
   clerrtop1 = 100,
   clerrtop5 = 100,
   epoch = 0
}

for epoch = 1, config.nepoch do
   print('Training ...')
   engine.mode = 'train'
   trainiter:exec('manualSeed', config.seed + epoch)
   trainiter:exec('resample')
   engine:train{
      maxepoch    = 1,
      network     = net,
      iterator    = trainiter,
      criterion   = criterion,
      optimMethod = optim.adam,
      config      = {
         learningRate      = config.lr,
         learningRateDecay = config.lrd
      },
   }
   print('Testing ...')
   engine.mode = 'val'
   engine:test{
      network   = net,
      iterator  = valiter,
      criterion = criterion,
   }
   if bestepoch.clerrtop1 > meter.clerr:value{k = 1} then
      bestepoch = {
         clerrtop1 = meter.clerr:value{k = 1},
         clerrtop5 = meter.clerr:value{k = 5},
         epoch = epoch,
         confm = meter.confm:value():clone()
      }
      torch.save(pathbestepoch, bestepoch)
      torch.save(pathbestnet, net:clearState())
   end
end
