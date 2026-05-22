"""
图像识别引擎模块：基于 PyTorch 加载 MobileNetV2 预训练模型，
对输入图像进行预处理和推理，返回 Top-K 分类结果及置信度。
所有推理均在 CPU 上完成，无需 GPU。
"""

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import os
import json
import urllib.request

# ---------- 常量 ----------

MODEL_CACHE = os.path.join(os.path.dirname(__file__), "model_cache")
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_CACHE = os.path.join(MODEL_CACHE, "imagenet_classes.txt")
ZH_MAP_CACHE = os.path.join(MODEL_CACHE, "imagenet_classes_zh.json")

# 常见 ImageNet 类别的中文翻译（覆盖日常识别中最高频的 200+ 类）
_LABEL_ZH_FALLBACK = {
    "tench": "丁鱥", "goldfish": "金鱼", "great white shark": "大白鲨",
    "tiger shark": "虎鲨", "hammerhead": "锤头鲨", "electric ray": "电鳐",
    "stingray": "黄貂鱼", "cock": "公鸡", "hen": "母鸡", "ostrich": "鸵鸟",
    "brambling": "燕雀", "goldfinch": "金翅雀", "house finch": "家朱雀",
    "junco": "灯心草雀", "indigo bunting": "靛蓝彩鹀", "robin": "知更鸟",
    "bulbul": "夜莺", "jay": "松鸦", "magpie": "喜鹊", "chickadee": "山雀",
    "water ouzel": "河乌", "kite": "鸢", "bald eagle": "白头海雕",
    "vulture": "秃鹫", "great grey owl": "乌林鸮", "salamander": "蝾螈",
    "newt": "蝾螈", "eft": "水蜥", "spotted salamander": "斑点钝口螈",
    "axolotl": "美西螈", "bullfrog": "牛蛙", "tree frog": "树蛙",
    "tailed frog": "尾蟾", "loggerhead": "蠵龟", "leatherback turtle": "棱皮龟",
    "mud turtle": "泥龟", "terrapin": "水龟", "box turtle": "箱龟",
    "banded gecko": "带纹壁虎", "common iguana": "美洲鬣蜥",
    "american chameleon": "美洲变色龙", "whiptail": "鞭尾蜥",
    "agama": "鬣蜥", "frilled lizard": "伞蜥", "alligator lizard": "钝尾毒蜥",
    "gila monster": "希拉毒蜥", "green lizard": "绿蜥蜴",
    "african chameleon": "非洲变色龙", "komodo dragon": "科莫多巨蜥",
    "triceratops": "三角龙", "thunder snake": "雷蛇", "ringneck snake": "环颈蛇",
    "hognose snake": "猪鼻蛇", "green snake": "翠青蛇", "king snake": "王蛇",
    "garter snake": "束带蛇", "water snake": "水蛇", "vine snake": "藤蛇",
    "night snake": "夜蛇", "boa constrictor": "红尾蚺", "rock python": "岩蟒",
    "indian cobra": "印度眼镜蛇", "green mamba": "绿曼巴蛇", "sea snake": "海蛇",
    "diamondback": "菱背响尾蛇", "sidewinder": "角响尾蛇", "trilobite": "三叶虫",
    "harvestman": "盲蛛", "scorpion": "蝎子", "tarantula": "狼蛛",
    "wolf spider": "狼蛛", "tick": "蜱虫", "centipede": "蜈蚣",
    "black grouse": "黑琴鸡", "ptarmigan": "雷鸟", "ruffed grouse": "披肩榛鸡",
    "prairie chicken": "草原松鸡", "peacock": "孔雀", "quail": "鹌鹑",
    "partridge": "山鹑", "african grey": "非洲灰鹦鹉", "macaw": "金刚鹦鹉",
    "cockatoo": "凤头鹦鹉", "lorikeet": "澳洲吸蜜鹦鹉", "coucal": "褐翅鸦鹃",
    "bee eater": "蜂虎", "hornbill": "犀鸟", "hummingbird": "蜂鸟",
    "jacamar": "鹟䴕", "toucan": "巨嘴鸟", "drake": "公鸭",
    "merganser": "秋沙鸭", "goose": "鹅", "black swan": "黑天鹅",
    "tusker": "公野猪", "echidna": "针鼹", "platypus": "鸭嘴兽",
    "wallaby": "沙袋鼠", "koala": "考拉", "wombat": "袋熊",
    "jellyfish": "水母", "sea anemone": "海葵", "brain coral": "脑珊瑚",
    "flatworm": "扁虫", "nematode": "线虫", "conch": "海螺",
    "snail": "蜗牛", "slug": "蛞蝓", "sea slug": "海蛞蝓",
    "chiton": "石鳖", "nautilus": "鹦鹉螺", "dungeness crab": "珍宝蟹",
    "rock crab": "岩蟹", "fiddler crab": "招潮蟹", "king crab": "帝王蟹",
    "american lobster": "美洲龙虾", "spiny lobster": "龙虾",
    "crayfish": "小龙虾", "hermit crab": "寄居蟹", "isopod": "等足类",
    "white stork": "白鹳", "black stork": "黑鹳", "spoonbill": "琵鹭",
    "flamingo": "火烈鸟", "little blue heron": "小蓝鹭", "american egret": "大白鹭",
    "bittern": "麻鳽", "crane": "鹤", "limpkin": "秧鹤", "european gallinule": "紫水鸡",
    "american coot": "美洲骨顶", "bustard": "鸨", "ruddy turnstone": "翻石鹬",
    "red-backed sandpiper": "黑腹滨鹬", "redshank": "红脚鹬", "dowitcher": "半蹼鹬",
    "oystercatcher": "蛎鹬", "pelican": "鹈鹕", "king penguin": "帝企鹅",
    "albatross": "信天翁", "grey whale": "灰鲸", "killer whale": "虎鲸",
    "dugong": "儒艮", "sea lion": "海狮", "chihuahua": "吉娃娃",
    "japanese spaniel": "日本狆", "maltese dog": "马尔济斯犬",
    "pekinese": "北京犬", "shih-tzu": "西施犬", "blenheim spaniel": "布伦海姆猎犬",
    "papillon": "蝴蝶犬", "toy terrier": "玩具梗", "rhodesian ridgeback": "罗得西亚脊背犬",
    "afghan hound": "阿富汗猎犬", "basset": "巴吉度猎犬", "beagle": "比格犬",
    "bloodhound": "寻血猎犬", "bluetick": "蓝斑猎犬", "black-and-tan coonhound": "黑褐猎浣熊犬",
    "walker hound": "沃克猎犬", "english foxhound": "英国猎狐犬",
    "redbone": "红骨猎浣熊犬", "borzoi": "俄罗斯猎狼犬", "irish wolfhound": "爱尔兰猎狼犬",
    "italian greyhound": "意大利灵缇", "whippet": "惠比特犬",
    "ibizan hound": "伊比赞猎犬", "norwegian elkhound": "挪威猎麋犬",
    "otterhound": "水獭猎犬", "saluki": "萨路基猎犬", "scottish deerhound": "苏格兰猎鹿犬",
    "weimaraner": "魏玛犬", "staffordshire bullterrier": "斯塔福郡斗牛梗",
    "american staffordshire terrier": "美国斯塔福梗",
    "bedlington terrier": "贝灵顿梗", "border terrier": "边境梗",
    "kerry blue terrier": "凯利蓝梗", "irish terrier": "爱尔兰梗",
    "norfolk terrier": "诺福克梗", "norwich terrier": "诺维奇梗",
    "yorkshire terrier": "约克夏梗", "wire-haired fox terrier": "硬毛猎狐梗",
    "lakeland terrier": "湖畔梗", "sealyham terrier": "西里汉梗",
    "airedale": "万能梗", "cairn": "凯恩梗", "australian terrier": "澳大利亚梗",
    "dandie dinmont": "丹迪丁蒙梗", "boston bull": "波士顿梗",
    "miniature schnauzer": "迷你雪纳瑞", "giant schnauzer": "巨型雪纳瑞",
    "standard schnauzer": "标准雪纳瑞", "scotch terrier": "苏格兰梗",
    "tibetan terrier": "西藏梗", "silky terrier": "丝毛梗",
    "soft-coated wheaten terrier": "软毛麦色梗", "west highland white terrier": "西高地白梗",
    "lhasa": "拉萨犬", "flat-coated retriever": "平毛寻回犬",
    "curly-coated retriever": "卷毛寻回犬", "golden retriever": "金毛寻回犬",
    "labrador retriever": "拉布拉多犬", "chesapeake bay retriever": "切萨皮克湾寻回犬",
    "german short-haired pointer": "德国短毛指示犬", "vizsla": "维兹拉犬",
    "english setter": "英国雪达犬", "irish setter": "爱尔兰雪达犬",
    "gordon setter": "戈登雪达犬", "brittany spaniel": "布列塔尼犬",
    "clumber": "克伦伯猎鹬犬", "english springer": "英国激飞猎犬",
    "welsh springer spaniel": "威尔士激飞猎犬",
    "cocker spaniel": "可卡犬", "sussex spaniel": "苏塞克斯猎犬",
    "irish water spaniel": "爱尔兰水猎犬", "kuvasz": "库瓦兹犬",
    "schipperke": "史奇派克犬", "groenendael": "格罗安达犬",
    "malinois": "马林诺斯犬", "briard": "布里牧犬", "kelpie": "卡尔比犬",
    "komondor": "可蒙犬", "old english sheepdog": "英国古代牧羊犬",
    "shetland sheepdog": "喜乐蒂牧羊犬", "collie": "柯利牧羊犬",
    "border collie": "边境牧羊犬", "bouvier des flandres": "法兰德斯畜牧犬",
    "rottweiler": "罗威纳犬", "german shepherd": "德国牧羊犬",
    "doberman": "杜宾犬", "miniature pinscher": "迷你品",
    "greater swiss mountain dog": "大瑞士山地犬",
    "bernese mountain dog": "伯恩山犬", "appenzeller": "阿彭策尔山犬",
    "entlebucher": "恩特雷布赫山地犬", "boxer": "拳师犬",
    "bull mastiff": "斗牛獒", "tibetan mastiff": "藏獒",
    "french bulldog": "法国斗牛犬", "great dane": "大丹犬",
    "saint bernard": "圣伯纳犬", "eskimo dog": "爱斯基摩犬",
    "malamute": "阿拉斯加雪橇犬", "siberian husky": "西伯利亚哈士奇",
    "dalmatian": "斑点犬", "affenpinscher": "猴面犬",
    "basenji": "巴仙吉犬", "pug": "巴哥犬", "leonberg": "莱昂贝格犬",
    "newfoundland": "纽芬兰犬", "great pyrenees": "大白熊犬",
    "samoyed": "萨摩耶", "pomeranian": "博美犬", "chow": "松狮犬",
    "keeshond": "荷兰毛狮犬", "brabancon griffon": "布鲁塞尔格里芬犬",
    "pembroke": "彭布罗克威尔士柯基犬", "cardigan": "卡迪根威尔士柯基犬",
    "toy poodle": "玩具贵宾犬", "miniature poodle": "迷你贵宾犬",
    "standard poodle": "标准贵宾犬", "mexican hairless": "墨西哥无毛犬",
    "grey wolf": "灰狼", "white wolf": "白狼", "red wolf": "红狼",
    "coyote": "郊狼", "dingo": "澳洲野狗", "dhole": "豺",
    "african hunting dog": "非洲野犬", "hyena": "鬣狗", "red fox": "赤狐",
    "kit fox": "敏狐", "arctic fox": "北极狐", "grey fox": "灰狐",
    "tabby": "虎斑猫", "tiger cat": "虎猫", "persian cat": "波斯猫",
    "siamese cat": "暹罗猫", "egyptian cat": "埃及猫",
    "cougar": "美洲狮", "lynx": "猞猁", "leopard": "豹",
    "snow leopard": "雪豹", "jaguar": "美洲豹", "lion": "狮子",
    "tiger": "老虎", "cheetah": "猎豹", "brown bear": "棕熊",
    "american black bear": "美洲黑熊", "ice bear": "北极熊",
    "sloth bear": "懒熊", "mongoose": "猫鼬", "meerkat": "狐獴",
    "tiger beetle": "虎甲虫", "ladybug": "瓢虫", "ground beetle": "步甲",
    "long-horned beetle": "天牛", "leaf beetle": "叶甲虫",
    "dung beetle": "蜣螂", "rhinoceros beetle": "独角仙",
    "weevil": "象鼻虫", "fly": "苍蝇", "bee": "蜜蜂", "ant": "蚂蚁",
    "grasshopper": "蚱蜢", "cricket": "蟋蟀", "walking stick": "竹节虫",
    "cockroach": "蟑螂", "mantis": "螳螂", "cicada": "蝉",
    "leafhopper": "叶蝉", "lacewing": "草蛉", "dragonfly": "蜻蜓",
    "damselfly": "豆娘", "admiral": "蛱蝶", "ringlet": "小环蛱蝶",
    "monarch": "帝王蝶", "cabbage butterfly": "菜粉蝶",
    "sulphur butterfly": "黄粉蝶", "lycaenid": "灰蝶", "starfish": "海星",
    "sea urchin": "海胆", "sea cucumber": "海参", "wood rabbit": "棉尾兔",
    "hare": "野兔", "angora": "安哥拉兔", "hamster": "仓鼠",
    "porcupine": "豪猪", "fox squirrel": "狐松鼠", "marmot": "旱獭",
    "beaver": "河狸", "guinea pig": "豚鼠", "sorrel": "栗色马",
    "zebra": "斑马", "hog": "猪", "wild boar": "野猪", "warthog": "疣猪",
    "hippopotamus": "河马", "ox": "牛", "water buffalo": "水牛",
    "bison": "野牛", "ram": "公羊", "bighorn": "大角羊",
    "ibex": "野山羊", "hartebeest": "狷羚", "impala": "黑斑羚",
    "gazelle": "瞪羚", "arabian camel": "单峰骆驼",
    "llama": "羊驼", "weasel": "黄鼠狼", "mink": "水貂",
    "polecat": "艾鼬", "black-footed ferret": "黑足鼬", "otter": "水獭",
    "skunk": "臭鼬", "badger": "獾", "armadillo": "犰狳",
    "three-toed sloth": "三趾树懒", "orangutan": "猩猩",
    "gorilla": "大猩猩", "chimpanzee": "黑猩猩",
    "gibbon": "长臂猿", "siamang": "合趾猿", "guenon": "长尾猴",
    "patas": "赤猴", "baboon": "狒狒", "macaque": "猕猴",
    "langur": "叶猴", "colobus": "疣猴", "proboscis monkey": "长鼻猴",
    "marmoset": "狨猴", "capuchin": "卷尾猴", "howler monkey": "吼猴",
    "titi": "伶猴", "spider monkey": "蜘蛛猴", "squirrel monkey": "松鼠猴",
    "madagascar cat": "环尾狐猴", "indri": "大狐猴", "elephant": "大象",
    "indian elephant": "印度象", "warthog": "疣猪", "hippopotamus": "河马",
    "ox": "牛", "water buffalo": "水牛", "bison": "野牛",
    "ram": "公羊", "bighorn": "大角羊", "ibex": "野山羊",
    "hartebeest": "狷羚", "impala": "黑斑羚", "gazelle": "瞪羚",
    "arabian camel": "单峰骆驼", "llama": "羊驼",
    "weasel": "黄鼠狼", "mink": "水貂", "polecat": "艾鼬",
    "black-footed ferret": "黑足鼬", "otter": "水獭", "skunk": "臭鼬",
    "badger": "獾", "armadillo": "犰狳", "three-toed sloth": "三趾树懒",
    "orangutan": "猩猩", "gorilla": "大猩猩", "chimpanzee": "黑猩猩",
    "gibbon": "长臂猿", "siamang": "合趾猿", "guenon": "长尾猴",
    "patas": "赤猴", "baboon": "狒狒", "macaque": "猕猴",
    "langur": "叶猴", "colobus": "疣猴", "proboscis monkey": "长鼻猴",
    "marmoset": "狨猴", "capuchin": "卷尾猴", "howler monkey": "吼猴",
    "titi": "伶猴", "spider monkey": "蜘蛛猴", "squirrel monkey": "松鼠猴",
    "madagascar cat": "环尾狐猴", "indri": "大狐猴",
    "african elephant": "非洲象", "indian elephant": "印度象",
    "banana": "香蕉", "lemon": "柠檬", "orange": "橙子", "pineapple": "菠萝",
    "apple": "苹果", "granny smith": "青苹果", "strawberry": "草莓",
    "cherry": "樱桃", "fig": "无花果", "pomegranate": "石榴",
    "broccoli": "西兰花", "cauliflower": "花椰菜", "zucchini": "西葫芦",
    "spaghetti squash": "金丝瓜", "acorn squash": "橡子南瓜",
    "butternut squash": "冬南瓜", "cucumber": "黄瓜", "artichoke": "洋蓟",
    "bell pepper": "甜椒", "cardoon": "刺苞菜蓟", "mushroom": "蘑菇",
    "corn": "玉米", "tomato": "番茄", "espresso": "浓缩咖啡",
    "cup": "杯子", "wine bottle": "酒瓶", "beer bottle": "啤酒瓶",
    "beer glass": "啤酒杯", "cocktail shaker": "调酒器",
    "pizza": "披萨", "cheeseburger": "芝士汉堡", "hotdog": "热狗",
    "ice cream": "冰淇淋", "ice lolly": "冰棍", "bagel": "贝果",
    "pretzel": "椒盐卷饼", "burrito": "墨西哥卷饼", "carbonara": "培根蛋酱意面",
    "dough": "面团", "popcorn": "爆米花", "sushi": "寿司",
    "matchstick": "火柴", "lighter": "打火机", "candle": "蜡烛",
    "toilet tissue": "卫生纸", "soap": "肥皂", "sponge": "海绵",
    "television": "电视机", "monitor": "显示器", "laptop": "笔记本电脑",
    "keyboard": "键盘", "mouse": "鼠标", "printer": "打印机",
    "cellphone": "手机", "telephone": "电话", "clock": "时钟",
    "digital clock": "电子钟", "watch": "手表", "hourglass": "沙漏",
    "camera": "相机", "reflex camera": "单反相机", "microphone": "麦克风",
    "radio": "收音机", "loudspeaker": "扬声器", "headphones": "耳机",
    "guitar": "吉他", "piano": "钢琴", "violin": "小提琴",
    "cello": "大提琴", "harp": "竖琴", "flute": "长笛",
    "saxophone": "萨克斯", "trumpet": "小号", "drum": "鼓",
    "accordion": "手风琴", "harmonica": "口琴", "ocarina": "陶笛",
    "umbrella": "雨伞", "backpack": "背包", "purse": "钱包",
    "handkerchief": "手帕", "bicycle": "自行车", "car": "汽车",
    "sports car": "跑车", "bus": "公共汽车", "truck": "卡车",
    "police van": "警车", "ambulance": "救护车", "fire engine": "消防车",
    "motorcycle": "摩托车", "taxi": "出租车", "jeep": "吉普车",
    "airplane": "飞机", "airship": "飞艇", "helicopter": "直升机",
    "ship": "轮船", "sailboat": "帆船", "speedboat": "快艇",
    "canoe": "独木舟", "kayak": "皮划艇", "submarine": "潜艇",
    "train": "火车", "streetcar": "有轨电车", "trolleybus": "无轨电车",
    "school bus": "校车", "horse cart": "马车", "bullet train": "高铁",
    "suspension bridge": "悬索桥", "steel arch bridge": "钢拱桥",
    "fountain": "喷泉", "igloo": "冰屋", "castle": "城堡",
    "lighthouse": "灯塔", "skyscraper": "摩天大楼", "palace": "宫殿",
    "church": "教堂", "mosque": "清真寺", "temple": "寺庙",
    "barn": "谷仓", "greenhouse": "温室", "boathouse": "船库",
    "volcano": "火山", "cliff": "悬崖", "coral reef": "珊瑚礁",
    "seashore": "海滨", "sandbar": "沙洲", "desert": "沙漠",
    "mountain": "山", "valley": "山谷", "lake": "湖泊", "river": "河流",
    "waterfall": "瀑布", "rainbow": "彩虹", "sunset": "日落",
    "sunflower": "向日葵", "daisy": "雏菊", "rose": "玫瑰",
    "tulip": "郁金香", "orchid": "兰花", "poppy": "罂粟花",
    "lotus": "荷花", "jasmine": "茉莉花", "dandelion": "蒲公英",
}


class ImageRecognizer:
    """图像识别器：封装模型加载、图像预处理和 Top-K 推理。"""

    def __init__(self):
        os.makedirs(MODEL_CACHE, exist_ok=True)
        self._device = torch.device("cpu")
        self.model = None
        self._labels_en = []
        self._labels_zh = {}

    # ---------- 模型加载 ----------

    def load_model(self):
        """加载 MobileNetV2 预训练模型（CPU），置为评估模式。"""
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.model = mobilenet_v2(weights=weights)
        self.model.to(self._device)
        self.model.eval()

    # ---------- 标签加载 ----------

    def load_labels(self):
        """加载 ImageNet 1000 类英文标签，并合入中文翻译映射。"""
        # 加载英文标签（优先缓存，否则下载）
        if not os.path.exists(LABELS_CACHE):
            self._download_labels()
        with open(LABELS_CACHE, "r", encoding="utf-8") as f:
            self._labels_en = [line.strip() for line in f.readlines() if line.strip()]

        # 加载中文翻译（优先缓存，否则用内置映射）
        if os.path.exists(ZH_MAP_CACHE):
            with open(ZH_MAP_CACHE, "r", encoding="utf-8") as f:
                self._labels_zh = json.load(f)
        else:
            self._labels_zh = _LABEL_ZH_FALLBACK
            with open(ZH_MAP_CACHE, "w", encoding="utf-8") as f:
                json.dump(_LABEL_ZH_FALLBACK, f, ensure_ascii=False, indent=2)

    def _download_labels(self):
        """下载 ImageNet 类别标签文件到本地缓存。"""
        urllib.request.urlretrieve(LABELS_URL, LABELS_CACHE)

    # ---------- 图像预处理 ----------

    # 缓存的预处理变换（与模型训练时完全一致）
    _preprocess_fn = None

    @classmethod
    def _get_preprocess(cls):
        """获取与 MobileNetV2 训练时完全一致的预处理变换。"""
        if cls._preprocess_fn is None:
            cls._preprocess_fn = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
        return cls._preprocess_fn

    @classmethod
    def preprocess(cls, image_path):
        """
        加载并预处理图像，转换为模型输入张量。
        使用 MobileNetV2 官方训练预处理（Resize(232)→CenterCrop(224)→归一化）。
        返回 (PIL.Image, Tensor) —— 前者用于 GUI 展示，后者用于推理。
        """
        img = Image.open(image_path).convert("RGB")
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)

        preprocess_fn = cls._get_preprocess()
        input_tensor = preprocess_fn(img).unsqueeze(0)  # (1, 3, 224, 224)
        return img, input_tensor

    # ---------- 推理 ----------

    def predict(self, image_path, top_k=5):
        """
        对指定图像进行识别，返回 Top-K 预测结果。
        每项为 (排名, 类别中文名, 类别英文名, 置信度百分比) 元组。
        """
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")
        if not self._labels_en:
            raise RuntimeError("标签尚未加载，请先调用 load_labels()")

        display_img, input_tensor = self.preprocess(image_path)

        with torch.no_grad():
            input_tensor = input_tensor.to(self._device)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        topk_probs, topk_indices = torch.topk(probabilities, top_k)
        topk_probs = topk_probs.cpu().tolist()
        topk_indices = topk_indices.cpu().tolist()

        results = []
        for rank, (idx, prob) in enumerate(zip(topk_indices, topk_probs), start=1):
            if idx < len(self._labels_en):
                en_name = self._labels_en[idx]
            else:
                en_name = f"class_{idx}"
            zh_name = self._labels_zh.get(en_name, en_name)
            results.append((rank, zh_name, en_name, round(prob * 100, 2)))

        return results, display_img

    # ---------- 获取标签名 ----------

    def get_label_zh(self, idx):
        """按类别索引获取中文标签名。"""
        en = self._labels_en[idx] if 0 <= idx < len(self._labels_en) else ""
        return self._labels_zh.get(en, en)
