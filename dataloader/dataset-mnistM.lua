require 'torch'
require 'paths'

mnistM = {}

function mnistM.loadTrainSet(trainpath,maxLoad, geometry)
   return mnistM.loadDataset(trainpath, maxLoad, geometry)
end

function mnistM.loadTestSet(testpath,maxLoad, geometry)
   return mnistM.loadDataset(testpath, maxLoad, geometry)
end

function mnistM.loadDataset(fileName, maxLoad)
  -- mnist.download()

   local f = torch.load(fileName,'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   local nExample = f.data:size(1)
   print('nExample',nExample)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnistM> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<mnistM> done')

   local datasetM = {}
   datasetM.data = data
   datasetM.labels = labels

   function datasetM:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function datasetM:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function datasetM:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(datasetM, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
   end})

   return datasetM
end
