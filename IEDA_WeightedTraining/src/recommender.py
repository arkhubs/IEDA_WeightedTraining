import torch


class Recommender:
    """
    推荐器：支持多指标加权排序
    """
    def __init__(self, model, alpha=0.5, metric='hybrid'):
        self.model = model
        self.alpha = alpha
        self.metric = metric

    def recommend(self, features):
        with torch.no_grad():
            pred_click_logit, pred_play_time = self.model(features)
            click_prob = torch.sigmoid(pred_click_logit)
            if self.metric == 'click':
                score = click_prob
            elif self.metric == 'play_time':
                score = pred_play_time
            else:
                # hybrid: alpha*点击率 + (1-alpha)*播放时长
                score = self.alpha * click_prob + (1 - self.alpha) * pred_play_time
            recommended_idx = torch.argmax(score).item()
        return recommended_idx
