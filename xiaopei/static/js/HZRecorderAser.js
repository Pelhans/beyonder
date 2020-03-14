(function (window) {
    //兼容
    window.URL = window.URL || window.webkitURL;
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

    var HZRecorder = function (stream, config) {
        config = config || {};
        config.sampleBits = config.sampleBits || 16;      //采样数位 8, 16
        config.sampleRate = config.sampleRate || (44100 / 6);   //采样率(1/6 44100)

        var context = new (window.webkitAudioContext || window.AudioContext)(); // 创建一个音频环境对象
        var audioInput = context.createMediaStreamSource(stream); // 来关联可能来自本地计算机麦克风或其他来源的音频流MediaStream
        var createScript = context.createScriptProcessor || context.createJavaScriptNode; // 创建一个可以通过JavaScript直接处理音频的ScriptProcessorNode. 
        var recorder = createScript.apply(context, [4096, 1, 1]); // 第二个和第三个参数指的是输入和输出都是双声道。
        var silence_mean_value = 0; //记录静音均值2018.8.13
        var silence_list = [];  //2018.8.13
        var sliceBuffer_length = 0; //2018.8.13

        var audioData = {
             size: 0          //录音文件长度
            , buffer: []     //录音缓存 
            , bufferLenght: 0    //录音缓存 上次 已处理的 的长度 
            , inputSampleRate: context.sampleRate    //输入采样率
            , inputSampleBits: 16       //输入采样数位 8, 16
            , outputSampleRate: 8000    //输出采样率
            , oututSampleBits: 32       //输出采样数位 8, 16
            , lengthList: []

            , input: function (data) {
                this.buffer.push(new Float32Array(data));
                this.size += data.length;
            }
            , compress: function (Data) { //合并压缩
                var  _buffer=(Data=="")?this.buffer:Data;
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
                writeString('RIFF'); offset += 4;
                // 下个地址开始到文件尾总字节数,即文件大小-8 
                data.setUint32(offset, 36 + dataLength, true); offset += 4;
                // WAV文件标志
                writeString('WAVE'); offset += 4;
                // 波形格式标志 
                writeString('fmt '); offset += 4;
                // 过滤字节,一般为 0x10 = 16 
                data.setUint32(offset, 16, true); offset += 4;
                // 格式类别 (PCM形式采样数据) 
                data.setUint16(offset, 1, true); offset += 2;
                // 通道数 
                data.setUint16(offset, channelCount, true); offset += 2;
                // 采样率,每秒样本数,表示每个通道的播放速度 
                data.setUint32(offset, sampleRate, true); offset += 4;
                // 波形数据传输率 (每秒平均字节数) 单声道×每秒数据位数×每样本数据位/8 
                data.setUint32(offset, channelCount * sampleRate * (sampleBits / 8), true); offset += 4;
                // 快数据调整数 采样一次占用字节数 单声道×每样本的数据位数/8 
                data.setUint16(offset, channelCount * (sampleBits / 8), true); offset += 2;
                // 每样本数据位数 
                data.setUint16(offset, sampleBits, true); offset += 2;
                // 数据标识符 
                writeString('data'); offset += 4;
                // 采样数据总数,即数据总大小-44 
                data.setUint32(offset, dataLength, true); offset += 4;
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

                return new Blob([data], { type: 'audio/wav' });
            }
        };

        //开始录音
        this.start = function () {
            audioInput.connect(recorder);
            audioData.lengthList.length = 0;
            audioData.lengthList.push(0);
            audioData.bufferLenght = 0;
            recorder.connect(context.destination);// 连接到输出源
        }

        //停止
        this.stop = function () {
            audioData.bufferLenght=0;
            recorder.disconnect();
        }

        this.reStart = function () {
            recorder.connect(context.destination);
        }

        this.reStop = function () {
            recorder.disconnect();
        }

        //获取全部音频文件
        this.getBlob = function () {
            this.stop();
            return audioData.encodeWAV("");
        }
        //获取分段音频数据
        this.getSplitBlob = function () {
            var Data=audioData.buffer.slice(audioData.bufferLenght);//截取剩余录音
            audioData.bufferLenght=audioData.buffer.length;//赋值 已处理的 录音长
            return audioData.encodeWAV(Data);
        }

        //2018.8.13, updata audioData.lengthList when audio is active
        this.update_audioData_lengthlist = function () {
            audioData.lengthList.push(audioData.buffer.length - 5);
        }

        //获取分段音频数据
        this.getBatchSplitBlob = function (len) {
            var Data=audioData.buffer;//截取剩余录音
            return audioData.encodeWAV(Data);
        }

        //2018.8.13 新定义函数，用估计静音
        this.silence_estimate_HZ = function () {
            silence_list = audioData.buffer.slice(0, 5);
            var sum_silence = 0;
            for (var i=0; i< silence_list.length; i++){
                sum_silence =  sum_silence + HZRecorder.maxValue(silence_list[i]); // by max_value
//                sum_silence =sum_silence + HZRecorder.sumSingleBuffer(silence_list[i]); // by energy
            }
            silence_mean_value = 4 * (sum_silence / silence_list.length);  //energy value
            sliceBuffer_length = audioData.buffer.length;
        }

        //2018.8.13 新定义函数，用于实时语音的静音检测
        this.silence_detection_HZ = function () {
            var sum_test = 0;
            var audio_data_list = audioData.buffer.slice(sliceBuffer_length);
            for (var i=0; i < audio_data_list.length; i++){
                sum_test = sum_test + HZRecorder.maxValue(audio_data_list[i]) - silence_mean_value;
//                sum_test = sum_test + HZRecorder.sumSingleBuffer(audio_data_list[i]) - silence_mean_value;
            }
            sliceBuffer_length = audioData.buffer.length;
            if(sum_test > 0) { //effective voice
                return true; //energy
            }else{
                return false;
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
        throw new function () { this.toString = function () { return message; } }
    }
    //是否支持录音
    HZRecorder.canRecording = (navigator.getUserMedia != null);
    //获取录音机
    HZRecorder.get = function (callback, config) {
        if (callback) {
            if (navigator.getUserMedia) {
                navigator.getUserMedia(
                    { audio: true } //只启用音频
                    , function (stream) {
                        var rec = new HZRecorder(stream, config);
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
                HZRecorder.throwErr('当前浏览器不支持录音功能。'); return;
            }
        }
    }

    /*/2018.8.13 ENERGY: convert Float32array buffer to Int16Array and implement the sum operation, result is weak.
    HZRecorder.sumSingleBuffer = function(singleBuffer_data){
        var sum = 0;
        for (var i=0; i< singleBuffer_data.length; i++) {
            sum = sum + Math.abs(singleBuffer_data[i]); //将音频整数绝对值求对数
        }

        return Math.log(sum);
    } */

    //2018.8.13 maximum value, max or min
    HZRecorder.maxValue = function(singleBuffer_data) {
        var max = Math.max.apply(Math, singleBuffer_data);
        var min = Math.min.apply(Math, singleBuffer_data);
        if (max < Math.abs(min)) {
            max = Math.abs(min)
        }
        return max;
    }

    window.HZRecorder = HZRecorder;

})(window);
