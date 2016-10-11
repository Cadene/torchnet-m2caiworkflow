local WeldonAggregation, parent = torch.class('WeldonAggregation', 'nn.Module')

-- n: number of top instances
function WeldonAggregation:__init(nMaxInstances, nMinInstances)
  parent.__init(self)
  self.nMax = nMaxInstances
  self.nMin = nMinInstances or nMaxInstances

  self.maxIndices = torch.Tensor()
  self.minIndices = torch.Tensor()
end

function WeldonAggregation:updateOutput(input)
  local inputView
  if input:dim() == 3 then -- image
    inputView = input:view(1, input:size(1), input:size(2), input:size(3))
  elseif input:dim() == 4 then -- batch of images
    inputView = input
  else
    assert(false, 'error in WeldonAggregation:updateOutput, not a batch, not an image')
  end
  local batchSize = inputView:size(1)
  local nMap = inputView:size(2)
  local h = inputView:size(3)
  local w = inputView:size(4)

  local flatInfo = inputView:view(batchSize, nMap, h*w) -- flatten every spatial information
  local scores, indices = flatInfo:sort(flatInfo:dim(), true)

  self.maxIndices = indices[{{},{},{1,self.nMax}}]
  local minIdBegin = h*w - self.nMin + 1
  self.minIndices = indices[{{},{},{minIdBegin,h*w}}]

  local maxValues = scores[{{},{},{1,self.nMax}}] -- get top k Max instances
  local minValues = scores[{{},{},{minIdBegin,h*w}}]
  self.output = maxValues:sum(3):div(self.nMax) + minValues:sum(3):div(self.nMin)

  if input:dim() == 3 then
    self.output = self.output:view(nMap) -- keep only second dim (others are empty)
  elseif input:dim() == 4 then
    self.output = self.output:view(batchSize, nMap)
  end
  return self.output
end

function WeldonAggregation:updateGradInput(input, gradOutput)
  local inputView
  if input:dim() == 3 then -- image
    inputView = input:view(1, input:size(1), input:size(2), input:size(3))
  elseif input:dim() == 4 then -- batch of images
    inputView = input
  else
    assert(false, 'error in WeldonAggregation:updateGradInput, not a batch, not an image')
  end
  local batchSize = inputView:size(1)
  local nMap = inputView:size(2)
  local h = inputView:size(3)
  local w = inputView:size(4)

  local gradOutputView = gradOutput:view(batchSize, nMap, 1)
  local maxLossExp = torch.expand(gradOutputView, batchSize, nMap, self.nMax)
  local minLossExp = torch.expand(gradOutputView, batchSize, nMap, self.nMin)

  -- put the right loss at the right place using the indices
  self.gradInput:typeAs(inputView):resize(batchSize, nMap, h*w):zero()
  self.gradInput:scatter(3, self.maxIndices, maxLossExp)
  self.gradInput:scatter(3, self.minIndices, minLossExp)

  if input:dim() == 3 then
    self.gradInput = self.gradInput:view(nMap, h, w)
  elseif input:dim() == 4 then
    self.gradInput = self.gradInput:view(batchSize, nMap, h, w)
  end
  return self.gradInput
end

function WeldonAggregation:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.maxIndices:resize()
   self.maxIndices:storage():resize(0)
   self.minIndices:resize()
   self.minIndices:storage():resize(0)
end

function WeldonAggregation:__tostring__()
   local s =  string.format('%s(%d,%d)', torch.type(self), self.nMax, self.nMin)
   return s
end
