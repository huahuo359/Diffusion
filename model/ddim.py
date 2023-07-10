from model.ddpm import *


class DDIM(DDPM): 
    def __init__(self, 
                 network, 
                 steps=1000, 
                 beta_schedule = 'linear',
                 min_beta=0.0001, 
                 max_beta=0.02, 
                 device=None, 
                 img_size=(3, 64, 64)):
        super().__init__(
            network = network, 
            steps = steps, 
            beta_schedule = beta_schedule, 
            min_beta = min_beta, 
            max_beta = max_beta,
            device = device, 
            img_size = img_size 
        )
    
    # 使用 DDIM 的去噪过程,加速图像的生成
    def gen_img(self, 
                 xt, 
                 simple_var = True,
                 ddim_step = 100,  # 缩减生成的过程到100步
                 eta = 1
                 ):
        if simple_var:
            eta = 1 
        # 取出需要进行生成步骤的 time
        ti = torch.linspace(self.steps, 0, (ddim_step+1)).to(self.device).to(torch.long)
        
        n_samples = xt.shape[0]
        
        for i in range(1, ddim_step+1):
            cur_t = ti[i-1] - 1
            pre_t = ti[i] - 1
            abar_cur = self.alpha_bars[cur_t]
            abar_pre = self.alpha_bars[pre_t] if pre_t>=0 else 1
            # print(cur_t)
            eta_theta = self.network(xt, xt.new_full((n_samples,), cur_t.item(), dtype=torch.long)) # 预测出的可能添加的噪声
            var = eta * (1 - abar_pre)/(1 - abar_cur) * (1 - abar_cur/abar_pre)
            noise = torch.randn_like(xt) # 随机生成高斯噪声
            
            # DDIM 的去噪公式 
            first_item = (abar_pre/abar_cur)**0.5 * xt
            second_item = ((1 - abar_pre - var)**0.5 - (abar_pre*(1-abar_cur)/abar_cur)**0.5)*eta_theta
            
            if simple_var:
                third_iterm = (1 - abar_cur/abar_pre)**0.5 * noise
            else:
                third_iterm = var**0.5 * noise 
                
            xt = first_item + second_item + third_iterm 
        
        return xt