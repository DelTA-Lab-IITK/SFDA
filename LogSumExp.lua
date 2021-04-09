local LogSumExp, parent = torch.class('nn.LogSumExp', 'nn.Module')

function LogSumExp:__init()
  parent.__init(self)
end

function LogSumExp:updateOutput(input)
  self.output = input:clone()
  self.max_val = torch.max(self.output)

  -- print("MAXVAL:", self.max_val,  ' MINVAL:', torch.min(input))

  self.output = self.output - self.max_val
  self.output = torch.log(torch.sum(torch.exp(self.output), 2))

  self.output = self.output + self.max_val
  return self.output
end

function LogSumExp:updateGradInput(input, gradOutput)
  -- print("MAXVAL:", self.max_val,  ' MINVAL:', torch.min(input))
  self.gradInput = torch.Tensor(input:size()):cuda()
  local t1 = input - self.max_val
  local normalization = torch.sum(t1, 2)
  -- print('input',input:size())
  for i=1, input:size()[2] do
    local t2 = torch.reshape(torch.exp(t1[{{}, i}]), input:size()[1])
    -- print('t2',t2:size())
    -- print('torch.cdiv(t2, normalization)',torch.cdiv(t2, normalization):size())
    -- print('self.gradInput',self.gradInput:size())
    -- print('self.gradInput[{{}, i}]',self.gradInput[{{}, i}])
    -- print('self.gradInput[{{}, i}]',self.gradInput[{{}, i}]:size())

    self.gradInput[{{}, i}] = torch.cdiv(t2, normalization):cuda()
    self.gradInput[{{}, i}] = torch.cmul(self.gradInput[{{}, i}], gradOutput[{{}, i}]):cuda()
  end

  return self.gradInput
end
