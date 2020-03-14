(function (window) {
    //兼容
    window.URL = window.URL || window.webkitURL;
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

    var rec;//2018.9.16
    var count_0 = 0; //2018.10.8, 记录有效静音段数目
    var count_1 = 0; // 2018.10.8, 记录过渡段数目
    var status = 0;  // 记录端点检测激活状态，静音段用0表示；过渡段用1表示；语音段用2表示
    var record_data = 0; // 记录浏览器每次输入的数据
    var frame_counter = 0; // 记录浏览器输入数据的次数

    var silence_mean_energy = 0; //记录静音energy均值
    var E0_high_para = 6.7; // 设置较高的能量阈值参数，2018.10.9
    var E0_low_para = 4.3; // 设置较低的能量阈值参数，2018.10.9
    var E0_high = 0; //较高的能量阈值
    var E0_low = 0; //较低的能量阈值
    var E0_weight_para = 0.85; // 2018.10.22,设置动态阈值权值
    var zero_low = 150; //使用较低的过零率阈值，对应语音 2018.9.16
    var silence_list = [];  //2018.8.13,存储静音数据列表，用于估计静音和静音更新

    var triggered_0 = true;  //2018.8.13，用于静音触发
    var triggered_1 = false; //2018.10.8，用于判断status=1是否触发
    var triggered_2 = false;    //2018.8.13，用于语音status=2触发
    var triggered_enter = false;    //2018.8.13，用于静音触发换行
    var count_enter = 0; //2018.8.13,统计静音时间计数，用来换行
    var max_time = 180; //2018.9.16 设置最高时长,(n/10)s,直接返回最终结果，重新计数
    var min_speech = 5;  //2018.9.16 设置最短时长,(n/10)s,每段有效语音必须大于该时长

    //2018.10.29, 短句合并在一起发送去识别，这里面不包括换行静音
    var recognize_time = 22; //2018.10.29，发送语音进行识别的最短时长。若<2s，则认为是短句；否则，是可以发送的长句
    var recognize_combine_time = 56; //2018.10.29, 短句合并，发送合并短句进行识别的最短时长5s = n*0.1s。
    var triggered_combined = false; //2018.10.29, 是否触发短句合并，如果有效语音时长小于recognize_time,则触发，否则，不触发。
                                    // 如果触发，则保留最初始的起始点。不触发，则起始点继续往下更新，准备下一次的短句合并。
    var recognize_start = 0; //2018.10.29, 对于时长小于recognize_time的有效语音，记录最初短句的起始点start_point
    var recognize_end_last = 0; //2018.10.29, 对于时长小于recognize_time的有效语音，记录最初短句的上一个结束点recognize_end_last

    var enter_time = 20; //2018.9.16 设置换行时长(n/10)s
    var max_silence = 2; // 2018.10.8，设置最短静音时长
    var start_time = 4; // 语音起始时长(n/10)s
    var start_extend_time = 20; //首句：换行后第一句为首句，首句语音起始往前延伸2s，n*0.1s，注意不能超过上一句的末尾。
    var start_point = 0; //2018.10.8，语音起始索引位置
    var start_point_temp = 0; //2018.10.8，记录语音4s起始索引位置
    var end_point = 0; //2018.10.8，语音结束索引位置,用来指代上一句语音结束的索引位置
    var silence_frame_num = 5; //2018.9.6，语音起始用于静音值估计的静音帧
    var slice_silence_time = 15; //2018.1.29，用于记录静音silence_2s的截取长度。
    // 输出 energy 和 zero值到txt文件中
    var energy_list = []; //2018.10.22，将能量值存储到列表中
    var zero_list = []; //2018.10.22，将过零率存储到列表中

    var HZRecorder = function (stream, config) {
        config = config || {};
        config.sampleBits = config.sampleBits || 16;      //采样数位 8, 16
        config.sampleRate = config.sampleRate || (44100 / 6);   //采样率(1/6 44100)

        try {
            var context = new (window.webkitAudioContext || window.AudioContext)(); // 创建一个音频环境对象
        } catch (e) {
            console.log('!Your browser does not support AudioContext');
        }
        var audioInput = context.createMediaStreamSource(stream); // 来关联可能来自本地计算机麦克风或其他来源的音频流MediaStream
        var createScript = context.createScriptProcessor || context.createJavaScriptNode; // 创建一个可以通过JavaScript直接处理音频的ScriptProcessorNode.
        var recorder = createScript.apply(context, [4096, 1, 1]); // 第二个和第三个参数指的是输入和输出都是双声道。
        var isConnect = false; //2018.10.25 音频暂停

        var audioData = {
            size: 0          //录音文件长度
            , buffer: []     //录音缓存
            , bufferLenght: 0    //录音缓存 上次 已处理的 的长度
            , inputSampleRate: context.sampleRate    //输入采样率
            , inputSampleBits: 16       //输入采样数位 8, 16
            , outputSampleRate: 8000    //输出采样率
            , oututSampleBits: 32       //输出采样数位 8, 16

            , input: function (data) {
                if (isConnect) {
                    record_data = new Float32Array(data);
                    this.buffer.push(record_data);
                    this.size += data.length;
                    console.log("buffering：" + audioData.buffer.length);
                    HZRecorder.audio_detection(record_data); // 调用语音检测函数.
                }
            }
            , compress: function (Data) { //合并压缩
                var _buffer = (Data == "") ? this.buffer : Data;
                //合并
                allLen = 1;
                for (var i = 0; i < _buffer.length; i++) {
                    allLen = allLen + _buffer[i].length;
                }
                var data = new Float32Array(allLen);
                var offset = 0;
                for (var i = 0; i < _buffer.length; i++) {
                    data.set(_buffer[i], offset);
                    offset += _buffer[i].length;
                }
                //压缩
                var compression = parseInt(this.inputSampleRate / this.outputSampleRate);
                var length = data.length / compression;
                var result = new Float32Array(length);
                var index = 0, j = 0;
                while (index < length) {
                    result[index] = data[j];
                    j += compression;
                    index++;
                }
                return result;
            }
            , encodeWAV: function (Data) {
                var sampleRate = Math.min(this.inputSampleRate, this.outputSampleRate);
                var sampleBits = Math.min(this.inputSampleBits, this.oututSampleBits);
                var bytes = this.compress(Data);

                var dataLength = bytes.length * (sampleBits / 8);

                var buffer = new ArrayBuffer(44 + dataLength);
                var data = new DataView(buffer);

                var channelCount = 1;//单声道
                var offset = 0;

                var writeString = function (str) {
                    for (var i = 0; i < str.length; i++) {
                        data.setUint8(offset + i, str.charCodeAt(i));
                    }
                }

                // 资源交换文件标识符
                writeString('RIFF');
                offset += 4;
                // 下个地址开始到文件尾总字节数,即文件大小-8
                data.setUint32(offset, 36 + dataLength, true);
                offset += 4;
                // WAV文件标志
                writeString('WAVE');
                offset += 4;
                // 波形格式标志
                writeString('fmt ');
                offset += 4;
                // 过滤字节,一般为 0x10 = 16
                data.setUint32(offset, 16, true);
                offset += 4;
                // 格式类别 (PCM形式采样数据)
                data.setUint16(offset, 1, true);
                offset += 2;
                // 通道数
                data.setUint16(offset, channelCount, true);
                offset += 2;
                // 采样率,每秒样本数,表示每个通道的播放速度
                data.setUint32(offset, sampleRate, true);
                offset += 4;
                // 波形数据传输率 (每秒平均字节数) 单声道×每秒数据位数×每样本数据位/8
                data.setUint32(offset, channelCount * sampleRate * (sampleBits / 8), true);
                offset += 4;
                // 快数据调整数 采样一次占用字节数 单声道×每样本的数据位数/8
                data.setUint16(offset, channelCount * (sampleBits / 8), true);
                offset += 2;
                // 每样本数据位数
                data.setUint16(offset, sampleBits, true);
                offset += 2;
                // 数据标识符
                writeString('data');
                offset += 4;
                // 采样数据总数,即数据总大小-44
                data.setUint32(offset, dataLength, true);
                offset += 4;
                // 写入采样数据
                if (sampleBits === 8) {
                    for (var i = 0; i < bytes.length; i++, offset++) {
                        var s = Math.max(-1, Math.min(1, bytes[i]));
                        var val = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        val = parseInt(255 / (65535 / (val + 32768)));
                        data.setInt8(offset, val, true);
                    }
                } else {
                    for (var i = 0; i < bytes.length; i++, offset += 2) {
                        var s = Math.max(-1, Math.min(1, bytes[i]));
                        data.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                    }
                }

                return new Blob([data], {type: 'audio/wav'});
            }
        };

        //开始录音
        this.start = function () {
            count_1 = 0;
            audioInput.connect(recorder);
            audioData.bufferLenght = 0;
            isConnect = true;
            recorder.connect(context.destination);// 连接到输出源
        }

        //停止
        this.stop = function () {
            count_1 = count_1 + 1;
            audioData.bufferLenght = 0;
            recorder.disconnect();
            name = '我'
        }

        //重启会议
        this.reStart = function () {
            count_1 = 0;
            isConnect = true;

            audioInput.connect(recorder);
            recorder.connect(context.destination);
        }

        //暂停会议
        this.reStop = function () {
            count_1 = count_1 + 1;
            isConnect = false;
            //recorder.disconnect();
            audioInput.disconnect();
            //如果在短句合并过程中暂停会议，将短句发送去识别。
            if (triggered_combined) {
                HZRecorder.recognize(recognize_start, recognize_end_last);
            }
        }

        //获取全部音频文件
        this.getBlob = function () {
            this.stop();
            return audioData.encodeWAV("");
        }
        //获取分段音频数据
        this.getSplitBlob = function () {
            var Data = audioData.buffer.slice(audioData.bufferLenght);//截取剩余录音
            audioData.bufferLenght = audioData.buffer.length;//赋值 已处理的 录音长
            return audioData.encodeWAV(Data);
        }

        // 2018.10.9, 获取语音起始截取位置,语音起始截取位置= 起始点start_point - 起始点延伸时长start_extend_time"
        this.getStartSlice = function (start_point, end_point_last) {
            var start_slice_point = 0;
            //若起始点延伸没超过上一句end_point，起始截取索引位置start_slice= start_point - start_extend；否则，end_point.index + 1作为起始点
            if ((start_point - start_extend_time) > end_point_last) {
                console.log("语音起始切割位置：" + (start_point - start_extend_time));
                start_slice_point = start_point - start_extend_time; //语音起始点前移
            } else {
                console.log("语音起始切割位置：" + (end_point_last + 1));
                start_slice_point = end_point_last + 1;
            }
            return start_slice_point;
        }

        // 2018.10.29, 并返回截取音频
        this.getBatchSplitBlob = function (start_slice_point) {
            var Data = audioData.buffer.slice(start_slice_point);//截取剩余录音
            console.log("截取长度: " + Data.length);
            return audioData.encodeWAV(Data);
        }


        //获取buffer长度
        this.getBufferLength = function () {
            return audioData.buffer.length;
        }

        //2018.8.13 新定义函数，用估计静音。索引值= 长度-1
        this.silence_estimate_HZ = function (slice_start_frame = audioData.buffer.length - silence_frame_num - 1, slice_end_frame = audioData.buffer.length - 1) {
            if (audioData.buffer.length < silence_frame_num) {
                silence_list = audioData.buffer.slice(0);
            } else {
                silence_list = audioData.buffer.slice(slice_start_frame, slice_end_frame);
            }
            var sum_silence = 0;
            var zero_silence = 0;
            for (var i = 0; i < silence_list.length; i++) {
                sum_silence = sum_silence + HZRecorder.energyCompute(silence_list[i]); // by energy
                zero_silence = zero_silence + HZRecorder.zeroCompute(silence_list[i]);
            }
            silence_mean_energy = sum_silence / silence_list.length;  //energy value
            console.log("silence_mean_energy:" + silence_mean_energy);
            //2018.10.19,动态能量阈值
            if (E0_high == 0) {
                E0_high = E0_high_para * silence_mean_energy;
            } else {
                E0_high = E0_weight_para * E0_high + (1 - E0_weight_para) * (E0_high_para * silence_mean_energy);
            }
            if (E0_low == 0) {
                E0_low = E0_low_para * silence_mean_energy;
            } else {
                E0_low = E0_weight_para * E0_low + (1 - E0_weight_para) * (E0_low_para * silence_mean_energy);
            }
            console.log("较高固定静音能量阈值：" + E0_high);
            console.log("较低固定静音能量阈值：" + E0_low);
        }

        //2018.8.13 新定义函数，基于energy and zero to implement audio detection实时语音的静音检测
        this.silence_detection_HZ = function (record_data) {
            var energy = HZRecorder.energyCompute(record_data);
            var zero = HZRecorder.zeroCompute(record_data);
            // 记录端点检测激活状态，静音段用0表示；过渡段用1表示；语音段用2表示
            if ((zero <= zero_low) || (energy >= E0_high)) { //如果能量和过零率任一超过较高门限，则进入语音段
                console.log("语音段");
                return "2";
            } else if (E0_low < energy) { //如果能量和过零率任一超过低门限，则进入过渡段
                console.log("过渡段");
                return "1";
            } else {
                console.log("静音段");
                return "0";
            }

        }

        //回放
        this.play = function (audio) {
            audio.src = window.URL.createObjectURL(this.getBlob());
        }

        //上传
        this.upload = function (url, callback) {
            var fd = new FormData();
            fd.append("audioData", this.getBlob());
            var xhr = new XMLHttpRequest();
            if (callback) {
                xhr.upload.addEventListener("progress", function (e) {
                    callback('uploading', e);
                }, false);
                xhr.addEventListener("load", function (e) {
                    callback('ok', e);
                }, false);
                xhr.addEventListener("error", function (e) {
                    callback('error', e);
                }, false);
                xhr.addEventListener("abort", function (e) {
                    callback('cancel', e);
                }, false);
            }
            xhr.open("POST", url);
            xhr.send(fd);
        }

        //音频采集
        recorder.onaudioprocess = function (e) {
            audioData.input(e.inputBuffer.getChannelData(0));
            //record(e.inputBuffer.getChannelData(0));
        }

    };
    //抛出异常
    HZRecorder.throwError = function (message) {
        alert(message);
        throw new function () {
            this.toString = function () {
                return message;
            }
        }
    }
    //是否支持录音
    HZRecorder.canRecording = (navigator.getUserMedia != null);
    //获取录音机
    HZRecorder.get = function (callback, config) {
        if (callback) {
            if (navigator.getUserMedia) {
                navigator.getUserMedia(
                    {audio: true} //只启用音频
                    , function (stream) {
                        rec = new HZRecorder(stream, config);
                        callback(rec);
                    }
                    , function (error) {
                        switch (error.code || error.name) {
                            case 'PERMISSION_DENIED':
                            case 'PermissionDeniedError':
                                HZRecorder.throwError('用户拒绝提供信息。');
                                break;
                            case 'NOT_SUPPORTED_ERROR':
                            case 'NotSupportedError':
                                HZRecorder.throwError('浏览器不支持硬件设备。');
                                break;
                            case 'MANDATORY_UNSATISFIED_ERROR':
                            case 'MandatoryUnsatisfiedError':
                                HZRecorder.throwError('无法发现指定的硬件设备。');
                                break;
                            default:
                                result = "";
                                if (error.code) {
                                    result = result + error.code;
                                }
                                if (error.name) {
                                    result = result + error.name;
                                }
                                HZRecorder.throwError('无法打开麦克风。异常信息:' + result);
                                break;
                        }
                    });
            } else {
                HZRecorder.throwErr('当前浏览器不支持录音功能。');
                return;
            }
        }
    }

    //2018.8.13, compute energy
    HZRecorder.energyCompute = function (singleBuffer_data) {
        var sum = 0;
        for (var i = 0; i < singleBuffer_data.length; i++) {
            sum = sum + singleBuffer_data[i] * singleBuffer_data[i]; //将音频整数绝对值求对数
        }
        var energy_sqrt = Math.sqrt(sum);
        console.log(energy_sqrt);
//        energy_list.push(energy_sqrt);
        return energy_sqrt;
    }

    //2018.10.8, compute cross-zero
    HZRecorder.zeroCompute = function (record_data) {
        var sum = 0;
        for (var i = 0; i < record_data.length - 1; i++) {
            if (record_data[i] * record_data[i + 1] < 0) {
                sum = sum + 1;
            }
        }
        console.log("过零率：" + sum);
//        zero_list.push(sum);
        return sum;
    }

    //2018.8.13~define a method for silence detection
    HZRecorder.audio_detection = function (record_data) {
        frame_counter = frame_counter + 1
        if (frame_counter == silence_frame_num) {
            rec.silence_estimate_HZ(0, silence_frame_num);
        } else if (frame_counter > silence_frame_num) {
            status = rec.silence_detection_HZ(record_data); //静音检测
            switch (status) {
                case "2":
                    console.log("大王，羊来了");
                    count_1 = count_1 + 1;
                    count_0 = 0;
                    if (triggered_0 && (!triggered_1) && (!triggered_2)) {
                        triggered_0 = false;
                        start_point = rec.getBufferLength() - 1;//如果status由0到1，则需要先确定起始点,index = length -1
                    }
                    HZRecorder.speech_trigger();
                    if (triggered_enter) {
                        HZRecorder.return_func();
                    }
                    break;
                case "1":
                    count_1 = count_1 + 1;
                    count_0 = 0; //静音帧统计归零
                    if ((triggered_0) && (!triggered_1) && (!triggered_2)) {
                        triggered_1 = true;
                        triggered_0 = false;
                        start_point = rec.getBufferLength() - 1; //获取语音起始点
                    }
                    if (triggered_2) {
                        HZRecorder.speech_trigger();
                    }
                    if (triggered_enter) {
                        HZRecorder.return_func();
                    }
                    break;
                case "0":
                    count_0 = count_0 + 1;
                    if (!triggered_2) {  //判断count_0在status=2之前出现
                        count_1 = 0;
                        triggered_0 = true;
                        triggered_1 = false;
                    } else {  //count_0在status=2之后出现，triggered_2 = true
                        if (count_0 < max_silence) { //如果静音时长< 静音时长阈值，则继续录音
                            count_1 = count_1 + 1;
                        } else if (triggered_2) { // count_0 >= max_silence，触发静音
                            triggered_0 = true;
                            triggered_1 = false;
                            triggered_2 = false;
                            triggered_enter = true;
                            count_enter = max_silence - 1;
                            if (count_1 - max_silence >= min_speech) { //最小语音时长
                                console.log("silence：---------------------------");
                                //保存有效语音初始start_point,如果是断句triggered_combined = true，则不更新，否则更新。
                                console.log("start_point:" + start_point);
                                if (!triggered_combined) {
                                    recognize_start = start_point;
                                    recognize_end_last = end_point;
                                }
                                console.log("recognize_start:" + recognize_start);
                                //2018.10.29, 统计每段有效语音长度，如果length > recognize_time, 去识别；否则，进行短句合并，等下一段有效语音。
                                var speech_length = rec.getBufferLength() - start_point - 1;
                                console.log("speech_length:" + speech_length);
                                // 判断有效语音长度是否超过识别最短发送时长recognize_time
                                if (speech_length >= recognize_time) {
                                    //获取识别数据,recognize_start是识别起始点index位置，recognize_end_last是上一句识别的结束点index位置。
                                    HZRecorder.recognize(recognize_start, recognize_end_last);
                                } else {
                                    triggered_combined = true; //开始断句合并
                                    var recognize_combine_length = rec.getBufferLength() - recognize_start - 1;
                                    console.log("recognize_combine_length:" + recognize_combine_length);
                                    //判断短句合并的长度是否超过短句合并发送的最短时长recognize_combined_time
                                    if (recognize_combine_length >= recognize_combine_time) {
                                        HZRecorder.recognize(recognize_start, recognize_end_last);
                                    }
                                }
                                console.log("triggered_combined:" + triggered_combined);
                                end_point = rec.getBufferLength() - max_silence - 1; //索引index = length -1
                                console.log("end_point:" + end_point);
                                count_0 = 0;
                                count_1 = 0;
                            } else { //噪音
                                count_0 = 0;
                                count_1 = 0;
                            }
                        }
                    }
                    if (triggered_enter) {
                        HZRecorder.return_func();
                    }
                    break;
            }
        }
    }

    //判断是否触发语音，语音长度是否达到最大时长max_time和每4s发送一个临时识别结果。
    HZRecorder.speech_trigger = function () {
        if (count_1 >= start_time) {
            console.log("语音开始");
            triggered_2 = true;
            triggered_enter = false;
            if (count_1 >= max_time) { //如果连续说话超过最高时长，则直接返回最终结果
                console.log("counter_16s:" + count_1);
                var start_slice_point = rec.getStartSlice(start_point, end_point);
                var recognize_Data = rec.getBatchSplitBlob(start_slice_point);
                HZRecorder.analysis(recognize_Data, true);
                count_1 = 0;
                start_point = rec.getBufferLength(); //如果语音时长达到16s，则buffer下一帧的索引作为起始点，即start_point=(buffer.len+1) -1
            }
            else if (count_1 % 45 == 0) { //每4s返回一个临时结果
                var start_slice_point_temp = start_point + count_1 - 40;
                var start_slice_point_temp = rec.getStartSlice(start_point_temp, end_point);
                var recognize_Data_temp = rec.getBatchSplitBlob(start_slice_point_temp);
                HZRecorder.analysis(recognize_Data_temp, false);
            }
        }
    }

    // 换行函数
    HZRecorder.return_func = function () {
        count_enter = count_enter + 1;
        console.log("count_enter: " + count_enter);
        if (count_enter >= enter_time) { // 如果静音超过2秒，就换行
            triggered_enter = false;
            console.log("换行");
            console.log("重新进行静音能量值估计");
            rec.silence_estimate_HZ(); //如果遇到较长静音，则重新进行静音检测
            //HZRecorder.save_data();
            //更新silence_2s,silence_2s在识别前加到语音前后。
            var formData_return = new FormData();
            formData_return.append("return_silence", rec.getBatchSplitBlob(rec.getBufferLength() - slice_silence_time));
            HZRecorder.update_silence_2s(formData_return);
            // 如果在断句合并过程中出现换行静音，则直接送去识别
            if (triggered_combined) {
                var start_slice_point = rec.getStartSlice(recognize_start, recognize_end_last);
                var recognize_Data = rec.getBatchSplitBlob(start_slice_point);
                triggered_combined = false;
                var formData = new FormData();
                formData.append("file", recognize_Data);
                $.ajax({
                    url: "/record/analysis/",
                    type: "post",
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function (res) {
                        if (res["result"].length >= 2) { //20180813
                            // console.log(res['name'])
                            var picSrc = '/static/imge/user.png'
                            var photoSrc = '/static/photo/' + res['name'] + '.jpeg'
                            if (res['name'] !== name && res['name'] !== '' && res['name'] !== undefined) {
                                name = res['name']
                                white = white + "<br>" + '<img src="' + photoSrc + '" alt="' + res['name'] + '"' + ' style="margin-left: -8%;width: 6.5%"' + '>' + res['name'] + '说：' + "<br>" + res['result'] + "。" + "<br>"; //黑色
                                black = "";
                            } else {
                                white = white + res["result"] + "。" + "<br>"; //黑色
                                black = "";
                            }
                            document.getElementById("white").innerHTML = white; //document是javascript内置对象，即整个页面
                            document.getElementById("Rwhite").innerHTML = white; //document.getElementById("更改对象").innerHTML="更改文本"，更改一个链接的文本
                        }
                    },
                });
            } else {
                if (white.endsWith("，")) {
                    white = white.substring(0, white.length - 1) + "。" + "<br>"; // "<br>"标签在js中换行，实时显示在html中
                    document.getElementById("white").innerHTML = white; //document是javascript内置对象，即整个页面
                    document.getElementById("Rwhite").innerHTML = white; //document.getElementById("更改对象").innerHTML="更改文本"，更改一个链接的文本
                }
            }

        }
    }

    //用ajax将energy and zero数据保存到本地txt文件
    HZRecorder.save_data = function () {
        data = {
            "energy": energy_list,
            "zero": zero_list
        },
            $.ajax({
                type: "POST",
                url: "/record/save_data/",
                data: JSON.stringify(data),
                dataType: "json",
                success: function (result) {
                    console.log("输出数据到txt文件中");
                },
            });
    }

    //换行时，实时更新silence_2s,加到语音前后，用于语音识别。
    HZRecorder.update_silence_2s = function (formData_return) {
        $.ajax({
            url: "/record/update_silence/",
            type: "post",
            data: formData_return,
            processData: false,
            contentType: false,
            success: console.log("更新silence_2s"),
        });
    }

    // 识别函数，通过语音起始点、上一段语音结束点和语音识别接口组成。
    HZRecorder.recognize = function (start_point, end_point_last) {
        var start_slice_point = rec.getStartSlice(start_point, end_point_last);
        var recognize_Data = rec.getBatchSplitBlob(start_slice_point);
        HZRecorder.analysis(recognize_Data, true);
        triggered_combined = false;
    }
    // jquery调用语音识别接口
    var name = '我'
    HZRecorder.analysis = function (recognize_Data, flag) { //flag=true表示 为最终结果，flag=false表示临时结果
        // send audio data to recognize
        var formData = new FormData();
        formData.append("file", recognize_Data);
        $.ajax({
            url: "/record/analysis/",
            type: "post",
            data: formData,
            processData: false,
            contentType: false,
            success: function (res) {
                if (res["result"].length >= 2) { //20180813
                    console.log(res['name'])
                    console.log(res['result'])
                    var picSrc = '/static/imge/user.png'
                    var photoSrc = '/static/photo/' + res['name'] + '.jpeg'
                    if (res['name'] !== name && res['name'] !== '' && res['name'] !== undefined) {
                        name = res['name']
                        if (flag) {
                            white = white + "<br>" + '<img src="' + photoSrc + '" alt="' + res['name'] + '"' + ' style="margin-left: -8%;width: 6.5%"' + '>' + name + '说：' + "<br>" + res['result'] + "，"; //黑色
                            black = "";
                        } else {
                            black = black + res["result"] + "，"; //红色
                        }
                    } else {
                        if (flag) {
                            white = white + res["result"] + "，"; //黑色
                            black = "";
                        } else {
                            black = black + res["result"] + "，"; //红色
                        }
                    }
                }
                //展示结果
                document.getElementById("white").innerHTML = white;
                document.getElementById("Rwhite").innerHTML = white;
                document.getElementById("black").innerHTML = black;
                document.getElementById("Rblack").innerHTML = black;
                var ele = document.getElementById('allWord');
                ele.scrollTop = ele.scrollHeight;

                var smallele = document.getElementById('smallAllword');
                smallele.scrollTop = smallele.scrollHeight;
            },
        });
    }

    window.HZRecorder = HZRecorder;

})(window);
