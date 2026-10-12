# Phân tích lỗi thực nghiệm

Các ví dụ được lấy từ run 3.000 historical test sequences trên câu đã rút ngắn và nhãn tự động. Chọn ba trường hợp boundary/type mismatch cho mỗi loại. Ranh giới là chỉ số subword, không phải offset ký tự hoặc gold annotation.

## Color tại legacy row 7936

casual matte Claw adorbable hair clutchers 5 Price Rs. matte finish Plastic for Everyday hair Claw Clip keep in you want cute colour range grips hair with simplicity Takspin 5 Multicolor Design Matte finish plastic Sales clutcher Matte finish plastic

False positives (type, start, end): [('Color', 39, 41)]

False negatives (type, start, end): []

## Material tại legacy row 7936

casual matte Claw adorbable hair clutchers 5 Price Rs. matte finish Plastic for Everyday hair Claw Clip keep in you want cute colour range grips hair with simplicity Takspin 5 Multicolor Design Matte finish plastic Sales clutcher Matte finish plastic

False positives (type, start, end): [('Material', 1, 3)]

False negatives (type, start, end): []

## Weight tại legacy row 15813

Zovi 1,449 a everybody, exclusive Brown casual side. Leather and a Leather may vary size.Disclaimer may to material the designed are. Brown zipper on the sole. Leather upper and checkered for stylish Leather Weight shoe vary on color may due the material lighting Material Leather. Brown

False positives (type, start, end): [('Weight', 3, 7)]

False negatives (type, start, end): []

## Weight tại legacy row 3050

Occasion Black Bellies 399 you fine comfort. your but designed going look this pair season this make out this Bootwale Occasion 350 - vary Height 1.5 inch Black Bellies Material 350 per Shoe - the vary Black

False positives (type, start, end): [('Weight', 26, 27)]

False negatives (type, start, end): []

## Material tại legacy row 18928

Mens White Shirt Western Cotton ColorWhite Contents White Zobello has ultimate dandy contrast and look this been for that and Zobello Casual White Shirt General Occasion Casual Half Brand Slim Details Code machine at dry warm not in

False positives (type, start, end): [('Material', 5, 6)]

False negatives (type, start, end): []

## Material tại legacy row 4768

Silk delivery. Genuine Brown

False positives (type, start, end): [('Material', 0, 1)]

False negatives (type, start, end): []

## Color tại legacy row 5026

- Clear Genuine Guarantee. Clear

False positives (type, start, end): [('Color', 0, 1)]

False negatives (type, start, end): []

## Color tại legacy row 884

Ishin 499 Soft Western . in Georgette Soft

False positives (type, start, end): [('Color', 10, 11)]

False negatives (type, start, end): []

## Weight tại legacy row 884

Ishin 499 Soft Western . in Georgette Soft

False positives (type, start, end): [('Weight', 2, 4)]

False negatives (type, start, end): []

## Dimensions tại legacy row 3289

German silver Zircon Toe Price fashion with classy rings and outfits and toed shoes.,Specifications German Number Jewellery Toe Model Women Color Gold Silver Diamond Diamond 6 g Height 45 mm Body Design Material Ring Additional Features 6 g. Alloy. Free

False positives (type, start, end): [('Dimensions', 34, 37)]

False negatives (type, start, end): []

## Dimensions tại legacy row 8144

Canvas Canvas inch x General Brand Canvas Theme Weight 2 kg 16 Width 1 Weight 2 kg

False positives (type, start, end): [('Dimensions', 12, 13)]

False negatives (type, start, end): []

## Dimensions tại legacy row 967

Features Ceramic Pack 1 Mug,AKUP Ceramic Price Ceramic you your with of ceramic 300ml Specifications Product - 9.5 cms, 8 Color Multi-Color Specialty Wash Remove to then with pattern tone product slightly that shown in image. in quality. co,Specifications of AKUP crazy-women Ceramic General Capacity Material Ceramic Freezer Diameter 8 mm 330 g Height mm the Sales 1 Ceramic. Weight 330 g

False positives (type, start, end): [('Dimensions', 66, 69)]

False negatives (type, start, end): []

## Lỗi và giới hạn trên câu nguyên vẹn

Service smoke dự đoán Weight thành 2 kg. gồm dấu chấm cuối câu. Đây là lỗi boundary cụ thể, được giữ trong service_smoke.json thay vì sửa evidence. Một mô tả dài có thể được xử lý đủ tới cuối nhưng model vẫn bỏ sót thuộc tính; kiểm thử coverage không được diễn giải là chứng minh semantic accuracy.

Hướng kiểm chứng: annotator độc lập sửa span trên mô tả nguyên vẹn, không xem prediction trước; chốt hướng dẫn về punctuation, units, phủ định, multiword material và catalog noisy. Không khắc phục bằng cách báo cáo weighted token F1 như entity F1.