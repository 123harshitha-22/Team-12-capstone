# X-ray Pathway
xray_input = Input(shape=(224, 224, 3))
xray_features = DenseNet121(weights='imagenet', include_top=False)(xray_input)
xray_features = GlobalAveragePooling2D()(xray_features)

# CT Scan Pathway
ct_input = Input(shape=(128, 128, 64, 1))  # 64 slices of size 128x128
ct_features = ResNet3D(weights=None)(ct_input)  # Custom 3D ResNet
ct_features = GlobalAveragePooling3D()(ct_features)

# Feature Fusion
fused_features = Concatenate()([xray_features, ct_features])
fused_features = Dense(512, activation='relu')(fused_features)
fused_features = Dropout(0.5)(fused_features)

# Classification Head
output = Dense(num_classes, activation='softmax')(fused_features)

# Model
model = Model(inputs=[xray_input, ct_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
