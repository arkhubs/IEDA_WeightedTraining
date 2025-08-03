    def get_weight_dataloaders(self, batch_size=128, num_workers=4):
        """
        获取权重模型的训练和验证数据加载器
        
        Args:
            batch_size: 批次大小
            num_workers: 数据加载线程数
        
        Returns:
            训练和验证数据加载器的元组
        """
        # 生成训练数据
        train_features = []
        train_treatments = []
        train_labels = []
        
        # 从处理组采样
        n_treatment_samples = 1000  # 可以根据需要调整
        treatment_users = self.get_treatment_users(n_treatment_samples)
        for user_id in treatment_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(1.0)  # 处理组标志
                label = self.get_ground_truth(user_id, video_id)
                
                train_features.append(features)
                train_treatments.append(treatment)
                train_labels.append(label)
        
        # 从对照组采样
        n_control_samples = 1000  # 可以根据需要调整
        control_users = self.get_control_users(n_control_samples)
        for user_id in control_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(0.0)  # 对照组标志
                label = self.get_ground_truth(user_id, video_id)
                
                train_features.append(features)
                train_treatments.append(treatment)
                train_labels.append(label)
        
        # 生成验证数据
        val_features = []
        val_treatments = []
        val_labels = []
        
        # 从验证集采样
        n_val_samples = 500  # 可以根据需要调整
        val_users = self.get_validation_users(n_val_samples)
        for user_id in val_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                # 随机分配处理/对照标志，因为验证集用户可能在任一组
                treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
                label = self.get_ground_truth(user_id, video_id)
                
                val_features.append(features)
                val_treatments.append(treatment)
                val_labels.append(label)
        
        # 创建数据集
        train_dataset = WeightDataset(
            features=train_features,
            treatments=train_treatments,
            labels=train_labels
        )
        
        val_dataset = WeightDataset(
            features=val_features,
            treatments=val_treatments,
            labels=val_labels
        )
        
        # 创建数据加载器
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_dataloader, val_dataloader
    
    def get_prediction_dataloaders(self, batch_size=128, num_workers=4, is_test=False):
        """
        获取预测模型的训练和验证/测试数据加载器
        
        Args:
            batch_size: 批次大小
            num_workers: 数据加载线程数
            is_test: 是否返回测试集（而非验证集）
        
        Returns:
            训练和验证/测试数据加载器的元组
        """
        # 这个方法类似于get_weight_dataloaders，但是根据预测模型的需求构建数据
        # 可以采用类似的方式创建数据集和加载器
        
        # 生成训练数据
        train_features = []
        train_treatments = []
        train_labels = []
        
        # 从处理组和对照组采样
        n_train_samples = 2000  # 可以根据需要调整
        train_users = self.get_treatment_users(n_train_samples // 2) + self.get_control_users(n_train_samples // 2)
        for user_id in train_users:
            # 对每个用户生成几个样本
            videos = self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
                label = self.get_ground_truth(user_id, video_id)
                
                train_features.append(features)
                train_treatments.append(treatment)
                train_labels.append(label)
        
        # 创建训练数据集
        train_dataset = torch.utils.data.TensorDataset(
            torch.stack(train_features),
            torch.stack(train_treatments),
            torch.stack(train_labels)
        )
        
        # 创建训练数据加载器
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        # 选择验证集或测试集
        eval_users = self.get_validation_users(1000) if not is_test else []
        # 如果是测试集，从处理组和对照组选择未在训练中使用的用户
        if is_test:
            test_treatment_users = [uid for uid in list(self.treatment_users) if uid not in train_users][:500]
            test_control_users = [uid for uid in list(self.control_users) if uid not in train_users][:500]
            eval_users = test_treatment_users + test_control_users
        
        # 生成验证/测试数据
        eval_features = []
        eval_treatments = []
        eval_labels = []
        
        for user_id in eval_users:
            # 对每个用户生成几个样本
            videos = self.get_user_test_videos(user_id, 5) if is_test else self.get_user_train_videos(user_id, 5)
            for video_id in videos:
                # 生成特征
                features = self.create_features(user_id, video_id)
                treatment = torch.tensor(1.0) if user_id in self.treatment_users else torch.tensor(0.0)
                label = self.get_ground_truth(user_id, video_id)
                
                eval_features.append(features)
                eval_treatments.append(treatment)
                eval_labels.append(label)
        
        # 创建验证/测试数据集
        eval_dataset = torch.utils.data.TensorDataset(
            torch.stack(eval_features),
            torch.stack(eval_treatments),
            torch.stack(eval_labels)
        )
        
        # 创建验证/测试数据加载器
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_dataloader, eval_dataloader
