import os
import re
import shutil
import webcolors
import datetime
import json
import copy
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag, Comment, Declaration, NavigableString
from collections import deque, defaultdict



# ====================== Precompiled regular expressions ======================
STYLE_PROP_RE = re.compile(r'([a-zA-Z-]+)\s*:\s*([^;]+)', re.IGNORECASE)
COLOR_HEX_RE = re.compile(r'^#([0-9a-f]{3,8})$', re.IGNORECASE)
RGB_RE = re.compile(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,?\s*([\d.]+)?\s*\)', re.IGNORECASE)
HSL_RE = re.compile(r'hsla?\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%\s*,?\s*([\d.]+)?\s*\)', re.IGNORECASE)
LENGTH_RE = re.compile(r'([-+]?[\d.]+)(px|pt|em|rem|%)?')

# Added regular expression
CONDITIONAL_START_RE = re.compile(r'<!$$if\s+([^>]+)$$>', re.IGNORECASE)

CONDITIONAL_END_RE = re.compile(
    r'<!$$endif$$>',
    re.IGNORECASE
)

# Added regular expression for handling split-style comments
SPLIT_CONDITIONAL_RE = re.compile(
    r'<!$$if\s+([^>]+)$$>([\s\S]*?)<!$$endif$$>',
    re.IGNORECASE
)


CSS_COLOR_TOKEN_RE = re.compile(
    r'(#[0-9a-f]{3,8}\b|rgba?\([^)]+\)|hsla?\([^)]+\)|\b[a-z]+\b)',
    re.IGNORECASE
)

# ====================== Constant definitions ======================
INHERITABLE_PROPS = {
    'color', 'font-family', 'font-size', 'font-weight', 
    'font-style', 'line-height', 'text-indent', 'visibility',
    'opacity', 'background-color', 'bgcolor' , 'background'
} # Only properties placed in this set will be inherited

DEFAULT_STYLE = {
    'color': {'rgb': (0, 0, 0), 'alpha': 1.0},
    'background-color': {'rgb': (255, 255, 255), 'alpha': 1.0},
    'font-size': '16px',
    'opacity': '1'
}

EXCLUDED_TAGS = {'script', 'style', 'meta', 'link', 'noscript', 'svg', 'img', 'title'}

NON_STYLABLE_TAGS = EXCLUDED_TAGS | {'br', 'hr', 'img', 'meta', 'link'}

# ====================== DOM Analyzer ======================
class DOMAnalyzer:
    def __init__(self, html_content):
        # self.soup = BeautifulSoup(html_content, 'html.parser')
        self.soup = self._handle_split_conditionals(html_content)        
        self.class_styles = self._parse_css_class_styles()
        self._prune_tree()  # Added preprocessing step
        self.paths = []
    
    # ====================== Preprocessing ======================
    
    def analyze_dom_order(self):
        results = []
        file_visible = True

        def walk(node, inherited_style):
            nonlocal file_visible

            if isinstance(node, NavigableString) and not isinstance(node, (Comment, Declaration)):
                text = str(node)

                if text.strip() == "" and "\n" in text:
                    return

                if text:
                    visible, reasons = self._check_visibility(inherited_style)
                    if not visible:
                        file_visible = False

                    results.append({
                        "text": text,
                        "type": "text",
                        "style": copy.deepcopy(inherited_style),
                        "visible": visible,
                        "hidden_reasons": reasons if not visible else []
                    })
                return

            if not isinstance(node, Tag):
                return

            if not self._is_valid_node(node):
                return

            current_style = self._parse_node_style(node, inherited_style)

            # if node.name == "a":
            #     text = self._get_link_text(node)
            #     visible, reasons = self._check_visibility(current_style)

            #     results.append({
            #         "text": text,
            #         "type": "link",
            #         "url": node.get("href", ""),
            #         "style": copy.deepcopy(current_style),
            #         "visible": visible,
            #         "hidden_reasons": reasons if not visible else []
            #     })

            #     print("ANCHOR STYLE:", current_style)

            #     for child in node.descendants:
            #         if isinstance(child, Tag):
            #             child_style = self._parse_node_style(
            #                 child,
            #                 copy.deepcopy(current_style)
            #             )

            #             print(
            #                 "CHILD:",
            #                 child.name,
            #                 child_style.get("color")
            #             )

            #     return
          
            #  if node.name == "a":
            #  current_style["_link_url"] = node.get("href", "")

            if node.name == "a":
                text = self._get_link_text(node)

                visible, reasons = self._check_link_visibility(node, current_style)

                results.append({
                    "text": text,
                    "type": "link",
                    "url": node.get("href", ""),
                    "style": copy.deepcopy(current_style),  # keep anchor-level style
                    "visible": visible,
                    "hidden_reasons": reasons if not visible else []
                })

                return


            child_inherited_style = {
                k: copy.deepcopy(v)
                for k, v in current_style.items()
                if k in INHERITABLE_PROPS
            }

            # keep non-inherited hiding properties active for descendants
            for prop in [
                "display", "visibility", "opacity",
                "position", "left", "right", "top", "bottom",
                "clip", "clip-path", "filter",
                "width", "height",
                "min-width", "min-height",
                "max-width", "max-height",
                "overflow", "overflow-x", "overflow-y",
                "white-space",
            ]:
                if prop in current_style:
                    child_inherited_style[prop] = copy.deepcopy(current_style[prop])

            for child in node.children:
                walk(child, child_inherited_style)

        root = self.soup.find("body") or self.soup
        walk(root, copy.deepcopy(DEFAULT_STYLE))

        results.extend(self._process_comments())
        return results, file_visible




    def _parse_css_class_styles(self):
        """Parse simple CSS class rules from <style> blocks."""
        class_styles = {}

        for style_tag in self.soup.find_all("style"):
            css = style_tag.get_text()

            rules = re.findall(r'\.([a-zA-Z0-9_-]+)\s*\{([^}]*)\}', css)

            for class_name, style_body in rules:
                props = dict(STYLE_PROP_RE.findall(style_body))
                class_styles[class_name] = {
                    prop.lower(): value.strip()
                    for prop, value in props.items()
                }

        return class_styles

    def _prune_tree(self):
        """Preprocess DOM tree"""
        body = self.soup.find('body') or self.soup
        self._prune_empty_nodes(body)
    
    def _handle_split_conditionals(self, content):
        """Handle split-style conditional comments"""
        # Stage 1: merge split-style comments with the same condition
        condition_map = defaultdict(list)
        
        # Identify all conditional blocks
        matches = SPLIT_CONDITIONAL_RE.finditer(content)
        for match in matches:
            condition = match.group(1).strip()
            html_fragment = match.group(2)
            condition_map[condition].append(html_fragment)
        
        # Generate replacement content
        replacements = []
        for cond, fragments in condition_map.items():
            if len(fragments) > 1:
                # Merge split fragments
                merged_html = f'<!--[cond:{cond}]-->{"".join(fragments)}<!--[endcond]-->'
                for f in fragments:
                    pattern = re.escape(f'<![if {cond}]>{f}<!$$endif$$>')
                    replacements.append((pattern, merged_html))
        
        # Execute batch replacement
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Stage 2: regular comment handling
        return BeautifulSoup(content, 'html.parser')
        
    def _preprocess_conditional_comments(self, content):
        """Preprocess HTML content in conditional comments"""
        # Stage 1: normalize conditional comment markers
        content = CONDITIONAL_START_RE.sub(
            r'<!--[cond_begin:\1]-->',
            content
        )
        content = CONDITIONAL_END_RE.sub(
            '<!--[cond_end]-->',
            content
        )
        
        # Stage 2: build document tree and handle nested structures
        soup = BeautifulSoup(content, 'html.parser')
        stack = []
        
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            text = comment.strip()
            if text.startswith('[cond_begin:'):
                # Parse conditional expression
                condition = text[12:-3].strip()
                # Create virtual container
                container = soup.new_tag("div", **{
                    'class': 'conditional-container',
                    'data-condition': condition
                })
                # Replace comment node
                comment.replace_with(container)
                stack.append(container)
            elif text == '[cond_end]':
                if stack:
                    container = stack.pop()
                    # Close current container
                    comment.decompose()
                    # Handle nesting
                    if stack:
                        stack[-1].append(container)
            else:
                # Regular comment handling
                pass
                
        return soup
        
        
    def _process_comments(self):
        """Improved comment processing method"""
        results = []
        
        # Handle merged conditional comments
        for comment in self.soup.find_all(string=lambda t: isinstance(t, Comment)):
            text = comment.strip()
            
            # Handle merged conditional block
            if text.startswith('[cond:'):
                condition = text[6:-3].strip()
                parent = comment.parent
                if parent and parent.name == 'div' and 'conditional-container' in parent.get('class', []):
                    analyzer = DOMAnalyzer(str(parent))
                    analyzer.collect_paths()
                    elements, _ = analyzer.analyze_paths()
                    
                    results.extend({
                        **elem,
                        "type": "conditional_comment",
                        # "condition": condition,
                        "visible": False,
                        # "hidden_reasons": ["split_conditional"] + elem.get("hidden_reasons", [])
                    } for elem in elements)
            
            # Original regular comment handling
            else:
                results.append({
                    "text": text,
                    "type": "html_comment",
                    "visible": False,
                    # "hidden_reasons": ["html_comment"]
                })
        
        return results


    def _get_comment_path(self, comment):
        """Get the hierarchical path of the comment node"""
        path = []
        parent = comment.parent
        while parent and parent.name:
            if parent.name.startswith('!'):  # Handle conditional comment pseudo-node
                path.insert(0, parent.name.replace('!', '').split()[0])
            else:
                path.insert(0, parent.name)
            parent = parent.parent
        return path
    
    # ====================== Path handling ======================
    
    def _is_leaf(self, node):
        """Enhanced leaf-node determination"""
        if not isinstance(node, Tag):
            return False
        
        # Condition 1: the node itself has direct text
        has_self_text = any(
            isinstance(c, str) and c.strip()
            for c in node.contents
            if not isinstance(c, Tag)
        )
        
        # Condition 2: all child nodes are tags that cannot be styled
        all_children_unstylable = all(
            isinstance(c, Tag) and 
            c.name in NON_STYLABLE_TAGS 
            for c in node.children
        )
        
        # Condition 3: the node itself or child nodes contain tags with inheritable styles
        has_stylable_descendant = any(
            isinstance(c, Tag) and 
            c.name not in NON_STYLABLE_TAGS 
            for c in node.descendants
        )
        
        #return has_self_text or (all_children_unstylable and not has_stylable_descendant)
        return not any(
            isinstance(c, Tag) and self._is_valid_node(c)
            for c in node.children
        )
    
        # return has_self_text and not any(
        #     isinstance(c, Tag) and self._is_valid_node(c)
        #     for c in node.children
        # )
    
    def _is_empty_tag(self, tag):
        """Determine whether this is an empty tag with no content"""
        return len(tag.contents) == 0
    
    def _is_valid_node(self, node):
        """Determine whether the node is valid (a tag that needs processing)"""
        return (
            isinstance(node, Tag) and 
            node.name not in NON_STYLABLE_TAGS and
            not isinstance(node, (Comment, Declaration))
        )
    
    def _process_anchor_tag(self, node, inherited_style):
        """Dedicated handling for <a> tags"""
        text = self._get_link_text(node)
        if not text:
            return None
        
        # Parse link style
        link_style = self._parse_node_style(node, inherited_style)
        visible, reasons = self._check_visibility(link_style)
        
        return {
            "text": text,
            "type": "link",
            "url": node.get('href', ''),
            # "path": [n.name for n in self._get_tag_path(node)],
            "style": link_style,
            "visible": visible,
            "hidden_reasons": reasons if not visible else []
        }

    def _get_link_text(self, node):
        """Get link text (including all child text)"""
        return ' '.join(node.stripped_strings)

    def _get_tag_path(self, node):
        """Get the independent path of a single tag"""
        path = []
        while node and node.name:
            path.insert(0, node.name)
            node = node.parent
        return path

    def _has_loose_text(self, node):
        """Detect whether the node contains loose text (direct text content)"""
        return any(
            isinstance(child, str) and child.strip()
            for child in node.contents
            if not isinstance(child, Tag)
        )
        
    def _prune_empty_nodes(self, node):
        """Post-order traversal to prune empty nodes"""
        if not isinstance(node, Tag):
            return False
        
        # Recursively process child nodes
        for child in list(node.children):  # Convert to list to avoid modification during traversal
            if self._prune_empty_nodes(child):
                child.decompose()  # Remove empty child node
        
        # Determine whether the current node is empty
        is_empty = (
            not self._has_any_text(node) and
            not self._has_meaningful_children(node)
        )
        return is_empty

    # def _has_any_text(self, node):
    #     """Check whether the node or descendants contain valid text"""
    #     return any(
    #         isinstance(c, str) and c.strip()
    #         for c in node.stripped_strings
    #     )

    def _has_any_text(self, node):
        """Check whether the node or descendants contain text, including intentional spaces."""
        return any(
            isinstance(c, NavigableString)
            and not isinstance(c, (Comment, Declaration))
            and str(c) != ""
            for c in node.descendants
        )

    def _has_meaningful_children(self, node):
        """Check whether there are valid child nodes"""
        return any(
            self._is_valid_node(child) 
            for child in node.children
            if isinstance(child, Tag)
        )

    def collect_paths(self):
        """Optimized path collection"""
        root = self.soup.find('body') or self.soup
        queue = deque([(root, tuple())])
        
        while queue:
            node, path = queue.popleft()
            # Skip comment nodes and non-tag nodes
            if not isinstance(node, Tag) or isinstance(node, (Comment, Declaration)):
                continue
            
            if not self._is_valid_node(node):  # Use the new method to validate
                continue
                
            new_path = path + (node,)
            
            if self._is_leaf(node):
                self.paths.append(new_path)
                continue
                
            # Add valid child nodes
            for child in node.children:
                if self._is_valid_node(child):
                    queue.append((child, new_path))
                    
    def analyze_paths(self):
        """Process style inheritance for all paths"""
        results = []
        file_visible = True
        seen_text_nodes = set() #new
        
        for path in self.paths:
            inherited_style = copy.deepcopy(DEFAULT_STYLE)
            
            for node in path:
                # ========== Added <a> tag handling logic ==========
                if node.name == 'a':
                    link_result = self._process_anchor_tag(node, inherited_style)
                    if link_result:
                        results.append(link_result)
                    # Skip subsequent child-node processing
                    break
                # ========== Original style handling logic ==========
                current_style = self._parse_node_style(node, inherited_style)
                #text_content = self._get_node_text(node)
                # is_last_node = node == path[-1]
                # text_content = self._get_node_text(node) if is_last_node else ""
                node_id = id(node)

                if node_id not in seen_text_nodes:
                    text_content = self._get_node_text(node)
                    seen_text_nodes.add(node_id)
                else:
                    text_content = ""

                if text_content:
                    visible, reasons = self._check_visibility(current_style)
                    if not visible:
                        file_visible = False
                    results.append({
                        "text": text_content,
                        "type": "text",  # Added text type identifier
                        # "path": [n.name for n in path],
                        "style": current_style,
                        "visible": visible,
                        "hidden_reasons": reasons if not visible else []
                    })
                
                # Prepare inherited properties
                inherited_style = {
                    k: copy.deepcopy(v) 
                    for k, v in current_style.items()
                    if k in INHERITABLE_PROPS
                }
                
        results.extend(self._process_comments())
        return results, file_visible

    # ====================== Style parsing ======================
    def _parse_opacity(self, current_value, parent_opacity='1'):
        """
        Parse opacity value and calculate cumulative opacity
        Parameters:
            current_value (str): opacity value of the current element (such as "0.5" or "50%")
            parent_opacity (str): opacity value of the parent element (default '1')
        Returns:
            str: calculated cumulative opacity string
        """
        # Parse parent element opacity
        try:
            parent_alpha = float(parent_opacity)
        except:
            parent_alpha = 1.0

        # Parse current element opacity
        try:
            # Remove spaces and handle percent sign
            value = current_value.strip().replace('%', '')
            current_alpha = float(value)
            
            # Handle percentage values
            if '%' in current_value:
                current_alpha /= 100.0
                
            # Clamp value range
            current_alpha = max(0.0, min(1.0, current_alpha))
        except:
            current_alpha = 1.0

        # Calculate cumulative opacity
        combined_alpha = parent_alpha * current_alpha
        return f"{combined_alpha:.2f}"

    def _parse_node_style(self, node, inherited_style):
        """Parse node style (including inheritance handling)"""
        # Deep-copy inherited style
        style = copy.deepcopy(inherited_style)
        
        # Handle element opacity attribute
        current_opacity = self._parse_opacity(
            node.get('opacity', '1'), 
            inherited_style.get('opacity', '1.0')
        )
        
        # Handle special tag attributes
        if node.name == 'font':
            attrs = {k.lower(): v for k, v in node.attrs.items()}
            if 'color' in attrs:
                #print("FONT COLOR ATTR:", repr(attrs["color"]))

                style['color'] = self._parse_legacy_html_color(attrs['color'])
                current_opacity = self._parse_opacity(style['color']['alpha'], current_opacity)
                
                #print("FONT COLOR PARSED:", style["color"])


            if 'size' in attrs:
                style['font-size'] = self._parse_font_size(attrs['size'])
                
        # ========== Added general bgcolor handling logic ==========
        if node.name in ('table', 'tr', 'td', 'th', 'font', 'body'):
            # Handle background color
            attrs = {k.lower(): v for k, v in node.attrs.items()}
            if 'bgcolor' in attrs:
                parsed_color = self._parse_color(attrs['bgcolor'], 'bg')
                # Blend the current background with the existing background (handle nesting)
                if 'background-color' in style:
                    existing_bg = copy.deepcopy(style['background-color'])
                    new_bg = self._blend_colors(parsed_color, existing_bg)
                    style['background-color'] = new_bg
                else:
                    style['background-color'] = parsed_color
            if 'text' in attrs:
                style['color'] = self._parse_color(attrs['text'])
                current_opacity = self._parse_opacity(style['color']['alpha'], current_opacity)
            
            # Handle alignment
            if 'align' in attrs:
                align_map = {
                    'left': 'left',
                    'right': 'right',
                    'center': 'center',
                    'middle': 'center',
                    'justify': 'justify'
                }
                style['text-align'] = align_map.get(attrs['align'].lower(), 'left')
            
            # Handle width (automatically add unit)
            if 'width' in attrs:
                width = attrs['width']
                if re.match(r'^\d+$', width):  # Add px automatically for pure numbers
                    style['width'] = f"{width}px"
                else:
                    style['width'] = width  # Preserve values with units
        # ========== End of table attribute handling ==========
        # Apply CSS class styles
        node_classes = node.get("class", [])
        for class_name in node_classes:
            if class_name in self.class_styles:
                for prop, value in self.class_styles[class_name].items():
                    prop = prop.lower()
                    if prop in ['background-color', 'bgcolor']:
                        parsed_bg = self._parse_color(value, 'bg')

                        if not self._is_transparent_color(parsed_bg):
                            style['background-color'] = parsed_bg

                    elif prop == 'background':
                        style['background'] = value

                        parsed_bg = self._extract_background_color(value)

                        if parsed_bg:
                            style['background-color'] = parsed_bg
                    elif prop == 'color':
                        style[prop] = self._parse_color(value)
                        current_opacity = self._parse_opacity(style['color']['alpha'], current_opacity)
                    elif prop == 'font-size':
                        style[prop] = self._parse_font_size(value)
                    elif prop == 'opacity':
                        current_opacity = self._parse_opacity(value, current_opacity)
                    else:
                        style[prop] = value






        # Parse inline style
        if 'style' in node.attrs:
            inline_style = dict(STYLE_PROP_RE.findall(node['style']))
            for prop, value in inline_style.items():
                prop = prop.lower()
                if prop in ['background-color', 'bgcolor']:
                    parsed_bg = self._parse_color(value, 'bg')

                    if not self._is_transparent_color(parsed_bg):
                        style['background-color'] = parsed_bg

                elif prop == 'background':
                    style['background'] = value

                    parsed_bg = self._extract_background_color(value)

                    if parsed_bg:
                        style['background-color'] = parsed_bg
                elif prop == 'color':
                    style[prop] = self._parse_color(value)
                    current_opacity = self._parse_opacity(style['color']['alpha'], current_opacity)
                elif prop == 'font-size':
                    style[prop] = self._parse_font_size(value)
                elif prop == 'opacity':
                    # Modified call method here
                    current_opacity = self._parse_opacity(value, current_opacity)
                else:
                    style[prop] = value
        
        # Calculate cumulative opacity
        style['opacity'] = current_opacity
        # style['opacity'] = self._calculate_opacity(style)
        return style

    # ====================== Visibility check ======================
    def _check_visibility(self, style):
        """Comprehensive visibility detection"""
        reasons = []
        
        # Transparent text color
        color_value = style.get("color")
        parsed_color = self._parse_color(color_value)

        if parsed_color.get("alpha", 1.0) <= 0.01:
            reasons.append("Color transparent: color:transparent")


        # Basic property detection
        if style.get('visibility') in ('hidden', 'collapse'):
            reasons.append('Visibility control:visibility')
        if style.get('display') == 'none':
            reasons.append('Visibility control:display:none')
        
        # Opacity detection
        if float(style.get('opacity', 1)) <= 0.01:
            reasons.append(f'Opacity: opacity:{style["opacity"]}')
        
        # # Zero-width / clipped inline text detection
        if (
            self._check_zero_area_clipping(style)
            or self._check_clipping(style)
        ):
            reasons.append("Clipping area: clipping")




        # if self._check_zero_area_clipping(style):
        #     reasons.append("Clipping area: clipping")  # TODO: split into css_zero_width_overflow later if needed
        #     #reasons.append("Zero-area clipped text: zero_area_overflow")
      
        #         # Clipping detection
        # # if self._check_clipping(style):
        # #     reasons.append('Clipping area: clipping')

        # Correct contrast detection logic
        final_fg = self._parse_color(style['color'])

        final_bg = self._resolve_effective_background(style)    

        # bg = self._parse_color(style.get('background-color'), 'bg')
        
        # # Recursively blend to get the actual displayed color
        # final_bg = self._parse_color(self._blend_colors(bg, DEFAULT_STYLE['background-color']))
        
        # Calculate final contrast
        contrast = self._calculate_contrast(final_fg, final_bg)
        
        if final_fg == final_bg:
            reasons.append('Same color: same_color')
        elif contrast < 1.05:
            reasons.append(f'Contrast: low_contrast({contrast})')
        
        
        # Position offset detection
        if self._check_position_offset(style):
            reasons.append('Position offset: position_offset')
        
        
        # Font size detection
        font_size = self._parse_length(style.get('font-size', '16px'))
        if font_size < 3:  # Adjusted from 1px to 6px
            reasons.append(f'Font too small: {font_size}px')
        
        # Filter detection
        if self._check_filter(style):
            reasons.append('Filter: filter_effect')
        
        return (len(reasons) == 0, reasons)

    def _check_link_visibility(self, anchor_node, anchor_style):
        """
        Evaluate visibility of rendered text inside <a>, while preserving
        anchor-level output style in JSONL.
        """

        text_runs = []

        def walk(node, inherited_style):
            if isinstance(node, NavigableString) and not isinstance(node, (Comment, Declaration)):
                text = str(node)

                if text.strip() == "" and "\n" in text:
                    return

                if text:
                    visible, reasons = self._check_visibility(inherited_style)
                    text_runs.append((text, visible, reasons, copy.deepcopy(inherited_style)))
                return

            if not isinstance(node, Tag):
                return

            if not self._is_valid_node(node):
                return

            current_style = self._parse_node_style(node, inherited_style)

            child_inherited_style = {
                k: copy.deepcopy(v)
                for k, v in current_style.items()
                if k in INHERITABLE_PROPS
            }

            for prop in [
                "display", "visibility", "opacity",
                "position", "left", "right", "top", "bottom",
                "clip", "clip-path", "filter",
                "width", "height",
                "min-width", "min-height",
                "max-width", "max-height",
                "overflow", "overflow-x", "overflow-y",
                "white-space",
            ]:
                if prop in current_style:
                    child_inherited_style[prop] = copy.deepcopy(current_style[prop])

            for child in node.children:
                walk(child, child_inherited_style)

        for child in anchor_node.children:
            walk(child, copy.deepcopy(anchor_style))

        if not text_runs:
            return self._check_visibility(anchor_style)

        visible_runs = [run for run in text_runs if run[1]]
        hidden_runs = [run for run in text_runs if not run[1]]

        # If any actual rendered text inside the link is visible,
        # the link as a whole should not be marked hidden.
        if visible_runs:
            return True, []

        # If all rendered text runs are hidden, preserve the reasons.
        reasons = []
        for _, _, run_reasons, _ in hidden_runs:
            for reason in run_reasons:
                if reason not in reasons:
                    reasons.append(reason)

        return False, reasons



    def _check_position_offset(self, style):
        """Position offset detection"""
        offset_props = {
            'left', 'right', 'top', 'bottom',
            'margin-left', 'margin-right',
            'margin-top', 'margin-bottom',
            'text-indent'
        }
        return any(
            self._is_large_offset(style.get(prop, '0'))
            for prop in offset_props
        )

    def _check_zero_area_clipping(self, style):
        """
        Detect text hidden by a clipped near-zero box.

        Catches:
        - inline hidden letters: width:0 + overflow:hidden
        - hidden containers: height:0 + overflow:hidden
        - max-width/max-height clipping variants
        """

        overflow = str(style.get("overflow", "")).lower()
        overflow_x = str(style.get("overflow-x", "")).lower()
        overflow_y = str(style.get("overflow-y", "")).lower()

        clips_x = overflow in ("hidden", "clip") or overflow_x in ("hidden", "clip")
        clips_y = overflow in ("hidden", "clip") or overflow_y in ("hidden", "clip")

        width = self._parse_length(style.get("width", "auto"))
        height = self._parse_length(style.get("height", "auto"))
        max_width = self._parse_length(style.get("max-width", "auto"))
        max_height = self._parse_length(style.get("max-height", "auto"))

        tiny_width = (
            width is not None and width <= 1
        ) or (
            max_width is not None and max_width <= 1
        )

        tiny_height = (
            height is not None and height <= 1
        ) or (
            max_height is not None and max_height <= 1
        )

        return (clips_x and tiny_width) or (clips_y and tiny_height)
    
    def _check_clipping(self, style):
        """Clipping area detection"""
        return any(
            prop in style and self._is_hidden_clip(style[prop])
            for prop in ['clip-path', 'clip']
        )

    def _check_font_size(self, style):
        """Font size detection"""
        font_size = self._parse_length(style.get('font-size', '16px'))
        return font_size and font_size < 1  # Treat <1px as hidden

    def _check_filter(self, style):
        """Filter effect detection"""
        if 'filter' not in style:
            return False
        return any(
            self._is_hiding_filter(f)
            for f in style['filter'].split()
        )

    # ====================== Helper methods ======================
    def _parse_length(self, value, base_size=16):
        """Parse CSS length value into pixels"""
        match = LENGTH_RE.match(str(value))
        if not match:
            return None
            
        num, unit = match.groups()
        num = float(num)
        
        if unit == 'pt':
            return num * 1.333
        if unit == 'em':
            return num * base_size
        if unit == 'rem':
            return num * 16  # Assume root font size is 16px
        if unit == '%':
            return num / 100 * base_size
        return num  # px unit or no unit

    def _is_large_offset(self, value, threshold=1000):
        """Determine whether this is a large offset value"""
        parsed = self._parse_length(value)
        return parsed and abs(parsed) > threshold

    # def _is_hidden_clip(self, value):
        """Determine whether the clipping value hides content"""
        value = value.lower()
        return any(
            pattern in value
            for pattern in ['inset(100%', 'rect(0', 'circle(0', 'polygon(0 0']
        )

    def _is_hidden_clip(self, value):
        """Determine whether the clipping value hides content."""
        value = str(value).lower().replace(" ", "")

        # clip: rect(top,right,bottom,left)
        rect_match = re.match(r"rect\(([^)]+)\)", value)
        if rect_match:
            parts = rect_match.group(1).split(",")

            if len(parts) == 4:
                top = self._parse_length(parts[0]) or 0
                right = self._parse_length(parts[1]) or 0
                bottom = self._parse_length(parts[2]) or 0
                left = self._parse_length(parts[3]) or 0

                width = right - left
                height = bottom - top

                return width <= 1 or height <= 1

        return any(
            pattern in value
            for pattern in [
                "inset(100%",
                "circle(0",
                "polygon(0 0",
                "polygon(0,0"
            ]
        )


    def _is_hiding_filter(self, filter_func):
        """Determine whether the filter causes hiding"""
        filter_func = filter_func.lower()
        if filter_func.startswith('opacity('):
            return float(filter_func[7:-1].strip('%')) <= 5
        if filter_func.startswith('blur('):
            blur_val = filter_func[5:-1]
            return self._parse_length(blur_val) > 10  # >10px blur
        return False

    def _calculate_opacity(self, style):
        """Calculate cumulative opacity"""
        current_opacity = float(style.get('opacity', 1.0))
        return str(current_opacity)

    
    # ====================== Color handling ======================


    def _is_transparent_color(self, color):
        try:
            return self._parse_color(color, 'bg').get('alpha', 1.0) <= 0.01
        except Exception:
            return True

    # def _extract_background_color(self, background_value):
        """
        Extract a representative visible color from CSS background shorthand.
        For gradients, use the first parseable color stop.
        """
        if not background_value:
            return None

        value = str(background_value).strip().lower()
        if value in ('none', 'transparent', 'inherit', 'initial', 'unset'):
            return None

        for token in CSS_COLOR_TOKEN_RE.findall(value):
            if token in {
                'linear-gradient', 'radial-gradient', 'repeating-linear-gradient',
                'to', 'top', 'bottom', 'left', 'right', 'center',
                'cover', 'contain', 'no-repeat', 'repeat', 'solid'
            }:
                continue

            try:
                parsed = self._parse_color(token, 'bg')
                if parsed.get('alpha', 1.0) > 0.01:
                    return parsed
            except Exception:
                pass

        return None

    def _extract_background_color(self, background_value):
        """
        Extract representative color from CSS background shorthand.
        Supports gradients.
        """
        if not background_value:
            return None

        value = str(background_value).strip().lower()

        if value in ('none', 'transparent', 'inherit', 'initial', 'unset'):
            return None

        # 1. hex colors
        hex_matches = re.findall(r'#[0-9a-f]{3,8}', value, re.IGNORECASE)
        for token in hex_matches:
            try:
                return self._parse_color(token, 'bg')
            except:
                pass

        # 2. rgb()/rgba()
        rgb_matches = re.findall(r'rgba?\([^)]+\)', value, re.IGNORECASE)
        for token in rgb_matches:
            try:
                return self._parse_color(token, 'bg')
            except:
                pass

        # 3. hsl()/hsla()
        hsl_matches = re.findall(r'hsla?\([^)]+\)', value, re.IGNORECASE)
        for token in hsl_matches:
            try:
                return self._parse_color(token, 'bg')
            except:
                pass

        # 4. named colors
        words = re.findall(r'\b[a-z]+\b', value)

        ignore = {
            'linear', 'gradient', 'radial', 'repeating',
            'top', 'bottom', 'left', 'right', 'center',
            'deg', 'transparent'
        }

        for word in words:
            if word in ignore:
                continue

            try:
                return self._parse_color(word, 'bg')
            except:
                pass

        return None

    



    def _resolve_effective_background(self, style):
        """
        Resolve effective background for same_color / low_contrast.
        In this detector, background is carried down as a painted ancestor
        background, not CSS inheritance.
        """
        bg = style.get('background-color')

        if bg and not self._is_transparent_color(bg):
            return self._blend_colors(bg, DEFAULT_STYLE['background-color'])

        shorthand_bg = self._extract_background_color(style.get('background'))
        if shorthand_bg and shorthand_bg.get('alpha', 1.0) > 0.01:
            return self._blend_colors(shorthand_bg, DEFAULT_STYLE['background-color'])

        return DEFAULT_STYLE['background-color']
        
        
        
    
    
    
    
    def _parse_color(self, value, colortype = 'fg'):
        """Color parsing method supporting multiple input formats"""
        # Type dispatch handling
        try:
            if isinstance(value, str):
                return self._parse_color_string(value)
            elif isinstance(value, tuple):
                return self._parse_color_tuple(value)
            elif isinstance(value, list):
                return self._parse_color_tuple(tuple(value))
            else:
                return self._parse_color_dict(value)
        except:    
            
            if colortype == 'fg': # Unknown type returns default font color black
                return {'rgb': (0, 0, 0), 'alpha': 1.0}
            else: # Unknown type returns default background color white
                return {'rgb': (255, 255, 255), 'alpha': 1.0}
    
    def _parse_legacy_html_color(self, value, colortype="fg"):
        """
        Parse legacy HTML color attributes using email-client-tolerant rules.

        Rationale:
        - Thunderbird rendered <font color="#0000"> as visible text.
        - Standard CSS Color Level 4 parsing interprets #0000 as
        #00000000 (fully transparent black).
        - Applying CSS RGBA semantics to legacy HTML color attributes
        caused false hidden-text detections (Email 362).

        TODO:
        - Evaluate whether the same normalization should be applied to
        other legacy HTML color attributes:
            * bgcolor="..."
            * body text="..."
        - Only extend this behavior if future samples demonstrate
        renderer mismatches for those attributes.
        """
        
        value = str(value).strip().lower()

        # Missing hash in legacy attrs:
        # bgcolor="000000" -> "#000000"
        if re.fullmatch(r"[0-9a-f]{3}|[0-9a-f]{6}", value):
            value = "#" + value

        # Thunderbird/email-client behavior for legacy attrs:
        # do not treat #RGBA / #RRGGBBAA as alpha syntax
        if re.fullmatch(r"#[0-9a-f]{4}", value):
            value = "#" + value[1:4]

        elif re.fullmatch(r"#[0-9a-f]{8}", value):
            value = "#" + value[1:7]

        return self._parse_color(value, colortype)
    
    def _parse_color_string(self, value):
        """Enhanced color parsing with alpha channel support"""
        value = value.strip().lower()
        
        # Handle transparent color
        if value == 'transparent':
            return {'rgb': (0, 0, 0), 'alpha': 0.0}
        
        # Hexadecimal format
        if hex_match := COLOR_HEX_RE.match(value):
            hex_val = hex_match.group(1)
            
            # Expand shorthand format
            if len(hex_val) in (3, 4):
                hex_val = ''.join(c*2 for c in hex_val)
            
            # Parse RGB and alpha channel
            rgb_part = hex_val[:6]
            alpha_hex = hex_val[6:] if len(hex_val) > 6 else ''
            # Handle cases with fewer than two digits; default ff (opaque)
            alpha_hex = (alpha_hex + 'ff')[:2]
            alpha = int(alpha_hex, 16) / 255.0
            
            return {
                'rgb': (
                    int(rgb_part[0:2], 16),
                    int(rgb_part[2:4], 16),
                    int(rgb_part[4:6], 16)
                ),
                'alpha': alpha
            }
            
        # RGB/RGBA format
        if rgb_match := RGB_RE.match(value):
            r, g, b = map(int, rgb_match.groups()[:3])
            alpha = float(rgb_match.group(4) or 1)
            return {
                'rgb': (r, g, b),
                'alpha': alpha  # Corrected variable name
            }
        
        # HSL/HSLA format
        if hsl_match := HSL_RE.match(value):
            h = int(hsl_match.group(1))
            s = int(hsl_match.group(2).replace('%', ''))
            l = int(hsl_match.group(3).replace('%', ''))
            alpha = float(hsl_match.group(4) or 1)
            return self._hsl_to_rgb(h, s, l, alpha)
        
        
        r, g, b = webcolors.name_to_rgb(value)
        return {'rgb': (r, g, b), 'alpha': 1.0}
        
        
    def _parse_color_tuple(self, value_tuple):
        """Handle tuple-type color values"""
        # Format validation
        if len(value_tuple) not in (3, 4):
            return {'rgb': (0, 0, 0), 'alpha': 1.0}
        
        # Extract RGB and alpha
        r, g, b = value_tuple[:3]
        alpha = value_tuple[3] if len(value_tuple) == 4 else 1.0
        
        # Value range check
        def clamp(x):
            return max(0, min(x, 255))
        
        return {
            'rgb': (clamp(r), clamp(g), clamp(b)),
            'alpha': max(0.0, min(alpha, 1.0))
        }

    def _parse_color_dict(self, value_dict):
        """Handle dictionary-type color values"""
        # Field validation
        required_keys = {'rgb', 'alpha'}
        if not all(k in value_dict for k in required_keys):
            return {'rgb': (0, 0, 0), 'alpha': 1.0}
        
        # Deep copy to prevent polluting original data
        return {
            'rgb': tuple(value_dict['rgb']),
            'alpha': float(value_dict['alpha'])
        }

    def _hsl_to_hex(self, h, s, l, alpha=1.0):
        """Convert HSL to hexadecimal"""
        h /= 360.0
        s /= 100.0
        l /= 100.0
        
        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                t += 1 if t < 0 else 0
                t -= 1 if t > 1 else 0
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        r, g, b = (int(round(x*255)) for x in (r, g, b))
        return f'#{r:02x}{g:02x}{b:02x}' + (f'{int(alpha*255):02x}' if alpha < 1 else '')
    
    def _hsl_to_rgb(self, h, s, l, alpha=1.0):
        """Convert HSL to hexadecimal"""
        h /= 360.0
        s /= 100.0
        l /= 100.0
        
        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                t += 1 if t < 0 else 0
                t -= 1 if t > 1 else 0
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        r, g, b = (int(round(x*255)) for x in (r, g, b))
        return {
            'rgb': (r, g, b),  # Integer tuple(0-255)
            'alpha': alpha         # Floating-point number(0.0-1.0)
        }
    
    def _blend_colors(self, fg, bg):
        """Recursively blend colors until opaque"""
        # Type check and conversion
        # print(f"Blend colors: foreground {fg}，background {bg}")
        
        fg = self._parse_color(fg) if not isinstance(fg, dict) else fg
        bg = self._parse_color(bg, 'bg') if not isinstance(bg, dict) else bg
        
        # Handle fully transparent case
        alpha = fg['alpha'] + bg['alpha'] * (1 - fg['alpha'])
        if fg['alpha'] == 0:
            #return {'rgb': (0, 0, 0), 'alpha': 0.0}
            return bg

        # Blending formula: result = fg * fg.alpha + bg * (1 - fg.alpha)
        blended_rgb = [
            int(min(255, fg['rgb'][i] * fg['alpha'] + bg['rgb'][i] * bg['alpha'] * (1 - fg['alpha'])))
            for i in range(3)
        ]
        blended_alpha = fg['alpha'] + bg['alpha'] * (1 - fg['alpha'])
        
        return {
            'rgb': tuple(blended_rgb),
            'alpha': blended_alpha
        }

    def _parse_font_size(self, value):
        """Font size parser"""
        value = str(value).strip().lower()
        
        # Traditional font size (1-7)
        if value.isdigit() and 1 <= int(value) <= 7:
            sizes = {1:10, 2:13, 3:16, 4:18, 5:24, 6:32, 7:48}
            return f"{sizes[int(value)]}px"
        
        # Modern units
        if match := re.match(r'^([\d.]+)(px|pt|em|%)$', value):
            num, unit = match.groups()
            num = float(num)
            if unit == 'pt':
                return f"{round(num * 1.333)}px"
            return f"{num}{unit}"
        
        return '16px'
    
    def _normalize_color(self, color):
        """Normalize color into RGBA tuple"""
        try:
            if color.startswith('rgba'):
                r, g, b, a = map(float, color[5:-1].split(','))
                return (int(r), int(g), int(b), a)
            return webcolors.hex_to_rgb(color) + (1.0,)
        except:
            return (0, 0, 0, 1.0)

    def _calculate_contrast(self, fg, bg):
        """WCAG 2.1 standard contrast calculation"""
        def linearize(c):
            c /= 255.0
            return c/12.92 if c <= 0.03928 else ((c + 0.055)/1.055)**2.4
        
        
        # # Blend final displayed color
        # final_fg = self._blend_colors(fg, bg)
        # final_bg = self._blend_colors(bg, DEFAULT_STYLE['background-color'])
        
        # Calculate relative luminance (remove extra 0.05)
        l1 = sum(coeff * linearize(c) for coeff, c in zip([0.2126, 0.7152, 0.0722], fg['rgb']))
        l2 = sum(coeff * linearize(c) for coeff, c in zip([0.2126, 0.7152, 0.0722], bg['rgb']))

        # Correctly apply contrast formula
        l1 += 0.05
        l2 += 0.05
        return round(max(l1, l2) / min(l1, l2), 2)

    # ====================== Other utility methods ======================
    # def _get_node_text(self, node):
    #     """Intelligent text collection method"""
    #     # Case 1: child nodes have no visibility-related styles → merge all text
    #     # if not self._children_have_visibility_style(node):
    #     #     return node.get_text(" ", strip=True)
        
    #     # Case 2: style controls exist → collect only direct text
    #     texts = []
    #     for child in node.children:
    #         # Handle regular text nodes (exclude comments/declarations)
    #         if isinstance(child, NavigableString) and not isinstance(child, (Comment, Declaration)):
    #             stripped = child.strip()
    #             if stripped:
    #                 texts.append(stripped)
    #         # Note: Tag-type nodes are not handled; they are recursively processed by the path collector
    #     return ' '.join(texts).strip()
        

    def _get_node_text(self, node):
        """Collect direct text while preserving intentional whitespace."""
        texts = []

        for child in node.children:
            if isinstance(child, NavigableString) and not isinstance(child, (Comment, Declaration)):
                text = str(child)

                # ignore formatting-only newlines
                if text.strip() == "" and "\n" in text:
                    continue

                texts.append(text)

        return ''.join(texts)

    def _children_have_visibility_style(self, node):
        """Accurately detect styles that affect text visibility (recursive depth-first)"""
        VISIBILITY_PROPS = {
            # Colors and background
            'color', 'background-color', 'background', 'bgcolor',
            # Font and text
            'font', 'font-size', 'font-family', 'font-weight', 'font-style',
            'line-height', 'letter-spacing', 'text-indent', 'text-shadow',
            # Display control
            'display', 'visibility', 'opacity', 'filter', 
            # Layout hiding
            'position', 'top', 'left', 'right', 'bottom', 
            'clip', 'clip-path', 'overflow', 'transform',
            # Blend mode
            'mix-blend-mode', 'isolation',
            # Special effects
            'box-shadow', 'backdrop-filter',
            # Experimental properties
            'contain', 'will-change'
        }
        
        for child in node.children:
            if not isinstance(child, Tag):
                continue
                
            # Check inline style
            if 'style' in child.attrs:
                # Extract all style properties (case-insensitive)
                style_props = {k.lower() for k, _ in STYLE_PROP_RE.findall(child['style'])}
                if style_props & VISIBILITY_PROPS:
                    return True
                    
            # Check native HTML attributes
            if self._check_html_visibility_attrs(child):
                return True
                
            # Recursively detect child nodes (depth-first)
            if self._children_have_visibility_style(child):
                return True
                
        return False

    def _check_html_visibility_attrs(self, node):
        """Detect native HTML visibility-related attributes"""
        attrs = {k.lower(): v for k, v in node.attrs.items()}
        
        # Generic attribute detection
        if 'hidden' in attrs:
            return True
        if 'bgcolor' in attrs and attrs['bgcolor'] != 'transparent':
            return True
            
        # Tag-specific attributes
        tag = node.name.lower()
        if tag == 'font' and any(a in attrs for a in ['color', 'size']):
            return True
        if tag == 'marquee' and 'behavior' in attrs:
            return True
        if tag in ('table', 'td', 'th') and 'background' in attrs:
            return True
            
        # Deprecated but still used attributes
        if any(attrs.get(a) for a in ['noshade', 'nowrap']):
            return True
            
        return False
    
# ====================== File processing functions ======================
def detect_invisible_html(input_path, output_path=None):
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        analyzer = DOMAnalyzer(f.read())
        results, file_visible = analyzer.analyze_dom_order()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return results, file_visible


def process_html_file(input_path, output_path):
    try:
        results, file_visible = detect_invisible_html(input_path, output_path)

        if file_visible:
            write_log(f"{input_path}\tTEXT_ALL_VISIBLE")
        else:
            os.makedirs("invisible_htmls", exist_ok=True)
            os.makedirs("invisible_jsons", exist_ok=True)

            shutil.copy2(input_path, "invisible_htmls")
            shutil.copy2(output_path, "invisible_jsons")
            write_log(f"{input_path}\tFOUND_INVISIBLE")

        return results, file_visible

    except Exception as e:
        print(f"{input_path} processing error,\nError reason: {e}")
        return [], True



# def process_html_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        
        try:
            analyzer = DOMAnalyzer(f.read())
            # # print(f"Processing: {os.path.basename(input_path)}")
            # analyzer.collect_paths()
            # # print(f"Number of valid paths: {len(analyzer.paths)}")
            # results, file_visible = analyzer.analyze_paths()

            results, file_visible = analyzer.analyze_dom_order()

            with open(output_path, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res)+"\n")
        
            if file_visible:
                write_log(f"{input_path}\tTEXT_ALL_VISIBLE")
            else:
                shutil.copy2(input_path, 'invisible_htmls')
                shutil.copy2(output_path, 'invisible_jsons')
                write_log(f"{input_path}\tFOUND_INVISIBLE")
            
        except Exception as e:
            print(f"{input_path}processing error,\n Error reason{e}")
    
    
    
    

def get_already_files(log_file):
    print(f'{str(datetime.datetime.now())}\tQuerying the list of already processed files')
    already_files = []
    
    if not os.path.exists(log_file):
        with open("html_invisible_log.txt", 'w') as f:
            pass
    
    with open(log_file, 'r') as f:
        for line in f.readlines():
            already_files.append(line.split('\t')[0])
    
    print(f'{str(datetime.datetime.now())}\tProcessed files[{len(already_files)}]')
    
    return already_files
    
def write_log(msg):
    with open("html_invisible_log.txt", 'a', encoding='utf-8') as f:
        f.write(msg + '\n')



# ====================== Main program ======================

if __name__ == '__main__':
    
    command = 'test'
    
    if command == 'run':
        OUTPUT_DIR = 'text_analysis'
        already_files = get_already_files("html_invisible_log.txt")
        
        for i in range(2023, 2025):
            year = str(i)
            
            if not os.path.exists(os.path.join('text_analysis', year)):
                os.makedirs(os.path.join('text_analysis', year))
        
            INPUT_DIR = 'htmls' + "/" + year
        
            for filename in tqdm(os.listdir(INPUT_DIR), desc=f"Checking visibility for {year}"):
                if filename.endswith('.html'):
                    basename, extension = os.path.splitext(filename)
                    input_path = os.path.join(INPUT_DIR, filename)
                    if input_path in already_files:
                        print("Already checked, skipping")
                        continue
                    
                    output_path = os.path.join(OUTPUT_DIR, year, f"{basename}.json")
                    process_html_file(input_path, output_path)
        # ====================== End of main program ======================            
    else:

    # # ====================== Test program ======================
        
        INPUT_DIR = "test_outputs_html"
        OUTPUT_DIR = "test_outputs_json_css"
        os.makedirs(OUTPUT_DIR, exist_ok=True)


        for filename in tqdm(os.listdir(INPUT_DIR), desc=f"[TESTING] Running tests..."):
            if filename.endswith('.html'):
                basename, extension = os.path.splitext(filename)
                input_path = os.path.join(INPUT_DIR, filename)
                output_path = os.path.join(OUTPUT_DIR, f"{basename}.json")
                process_html_file(input_path, output_path)

